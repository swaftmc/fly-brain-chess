"""Train the fly-brain chess agent with REINFORCE."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import chess
import chess.engine
import torch
import torch.nn.functional as F

try:
    torch.sparse.check_sparse_tensor_invariants(False)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_agent import ChessBrain
from chess_encoding import board_to_rates
from chess_moves import move_to_index, NUM_MOVES

DEFAULT_DATA = ROOT.parent / "data"


def make_engine(path: str | None):
    candidates = [path] if path else []
    candidates += ["stockfish", "/usr/games/stockfish", "/usr/bin/stockfish"]
    for cand in candidates:
        if not cand:
            continue
        try:
            return chess.engine.SimpleEngine.popen_uci(cand)
        except Exception:
            continue
    return None


def material_score(board: chess.Board) -> int:
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    score = 0
    for sq, piece in board.piece_map().items():
        v = vals.get(piece.piece_type, 0)
        score += v if piece.color == chess.WHITE else -v
    return score


def evaluate(board: chess.Board, engine, depth: int = 6) -> int:
    if engine is None:
        return material_score(board)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        s = info["score"].white().score(mate_score=100000)
        return 0 if s is None else int(s)
    except Exception:
        return material_score(board)


def shaped_reward(board_before, move, board_after, engine, agent_color, terminal_bonus=0.0):
    eb = evaluate(board_before, engine)
    ea = evaluate(board_after, engine)
    if agent_color == chess.WHITE:
        r = (ea - eb) / 100.0
    else:
        r = (eb - ea) / 100.0
    # terminal
    if board_after.is_game_over():
        res = board_after.result()
        if res == "1-0":
            win = agent_color == chess.WHITE
        elif res == "0-1":
            win = agent_color == chess.BLACK
        else:
            win = None
        if win is True:
            r += 1.0
        elif win is False:
            r -= 1.0
    r += terminal_bonus
    return float(r)


def opponent_move(board, engine, opponent_elo: int):
    if engine is not None:
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": opponent_elo})
        except Exception:
            pass
        res = engine.play(board, chess.engine.Limit(depth=8))
        if res.move is not None:
            board.push(res.move)
            return
    legal = list(board.legal_moves)
    if legal:
        board.push(random.choice(legal))


def train(args):
    device = args.device
    data_dir = Path(args.data_dir)
    comp_path = data_dir / "2025_Completeness_783.csv"
    conn_path = data_dir / "2025_Connectivity_783.parquet"
    wt_dir = data_dir

    engine = make_engine(args.stockfish)

    if args.train_backbone:
        print("WARNING: full connectome backprop is disabled on Colab/T4 "
              "(sparse gradient is dense and OOMs). Training readout/policy only.")
        args.train_backbone = False

    agent = ChessBrain(
        comp_path=str(comp_path),
        conn_path=str(conn_path),
        wt_dir=str(wt_dir),
        num_neurons=0,
        readout_size=args.readout_size,
        stim_rate=args.stim_rate,
        sim_ms=args.sim_ms,
        device=device,
        train_backbone=args.train_backbone,
    )
    agent.to(device)
    agent.set_train_backbone(args.train_backbone)

    # optimizers
    dense_params = [p for n, p in agent.named_parameters() if n != "weight_values" and p.requires_grad]
    opt_dense = torch.optim.Adam(dense_params, lr=args.lr) if dense_params else None
    opt_sparse = None
    if args.train_backbone:
        opt_sparse = torch.optim.SparseAdam([agent.weight_values], lr=args.lr * 0.1)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for game_idx in range(1, args.games + 1):
        board = chess.Board()
        agent_color = chess.WHITE
        rewards = []

        if opt_dense:
            opt_dense.zero_grad()
        if opt_sparse:
            opt_sparse.zero_grad()

        while not board.is_game_over() and board.fullmove_number <= args.max_moves:
            if board.turn == agent_color:
                rates = board_to_rates(board, agent.num_neurons, agent.stim_rate)
                if torch.cuda.is_available():
                    print(f"    [move {board.fullmove_number}] "
                          f"gpu_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                          f"gpu_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
                logits, _value = agent(rates)
                legal_moves = list(board.legal_moves)
                legal_idx = [move_to_index(m) for m in legal_moves]
                legal_logits = logits[0, legal_idx]
                dist = torch.distributions.Categorical(logits=legal_logits)
                choice = dist.sample()
                selected = legal_moves[choice.item()]
                log_prob = dist.log_prob(choice)
                entropy = dist.entropy()

                board_before = board.copy()
                board.push(selected)
                r = shaped_reward(board_before, selected, board, engine, agent_color)
                rewards.append(r)

                reward_t = torch.tensor(r, device=device, dtype=torch.float32)
                loss = -log_prob * reward_t - args.entropy * entropy
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                opponent_move(board, engine, args.opponent_elo)

        if not rewards:
            print(f"Game {game_idx}: no agent moves")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # update accumulated gradients from all moves
        if opt_dense:
            torch.nn.utils.clip_grad_norm_(dense_params, args.grad_clip)
            opt_dense.step()
        if opt_sparse:
            torch.nn.utils.clip_grad_norm_([agent.weight_values], args.grad_clip)
            opt_sparse.step()

        outcome = board.result() if board.is_game_over() else "*"
        print(f"Game {game_idx}/{args.games} outcome={outcome} moves={len(rewards)} "
              f"return={sum(rewards):.3f}")
        if torch.cuda.is_available():
            print(f"  post-game gpu_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                  f"gpu_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
            torch.cuda.empty_cache()

        if game_idx % args.save_every == 0:
            ckpt = {
                "model": agent.state_dict(),
                "config": {
                    "comp_path": str(comp_path),
                    "conn_path": str(conn_path),
                    "wt_dir": str(wt_dir),
                    "num_neurons": agent.num_neurons,
                    "readout_size": args.readout_size,
                    "stim_rate": args.stim_rate,
                    "sim_ms": args.sim_ms,
                    "train_backbone": args.train_backbone,
                },
            }
            torch.save(ckpt, ckpt_dir / f"chess_agent_game{game_idx}.pt")
            print(f"  saved checkpoint game {game_idx}")

    if engine is not None:
        engine.quit()
    print("Training complete")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sim-ms", type=float, default=50.0)
    p.add_argument("--stim-rate", type=float, default=200.0)
    p.add_argument("--readout-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max-moves", type=int, default=80)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--opponent-elo", type=int, default=800)
    p.add_argument("--stockfish", default=None)
    p.add_argument("--train-backbone", action="store_true")
    p.add_argument("--checkpoint-dir", default=str(ROOT.parent / "checkpoints"))
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
