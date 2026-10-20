"""Evaluate a trained fly-brain chess agent against Stockfish."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_agent import load_checkpoint
from chess_encoding import board_to_rates
from chess_moves import move_to_index
import torch


def make_engine(path):
    for cand in ([path] if path else []) + ["stockfish", "/usr/games/stockfish", "/usr/bin/stockfish"]:
        if not cand:
            continue
        try:
            return chess.engine.SimpleEngine.popen_uci(cand)
        except Exception:
            continue
    return None


def choose_move(agent, board, device, temperature=0.1):
    rates = board_to_rates(board, agent.num_neurons, agent.stim_rate)
    logits, _ = agent(rates.to(device))
    legal_moves = list(board.legal_moves)
    legal_idx = [move_to_index(m) for m in legal_moves]
    legal_logits = logits[0, legal_idx] / max(temperature, 1e-6)
    probs = torch.softmax(legal_logits, dim=-1)
    return legal_moves[int(torch.argmax(probs).item())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--opponent-elo", type=int, default=800)
    p.add_argument("--stockfish", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    agent = load_checkpoint(args.checkpoint, device=args.device)
    agent.eval()
    engine = make_engine(args.stockfish)
    if engine is None:
        print("Stockfish not found; aborting")
        return
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.opponent_elo})
    except Exception:
        pass

    score = {"win": 0, "draw": 0, "loss": 0}
    for g in range(args.games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = choose_move(agent, board, args.device)
            else:
                res = engine.play(board, chess.engine.Limit(depth=8))
                move = res.move
            board.push(move)
        res = board.result()
        if res == "1-0":
            score["win"] += 1
        elif res == "0-1":
            score["loss"] += 1
        else:
            score["draw"] += 1
        print(f"Game {g+1}: {res}")
    print("Summary:", score)
    engine.quit()


if __name__ == "__main__":
    main()
