"""Play chess with a trained fly-brain agent."""

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


def make_engine(path):
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


def choose_move(agent, board, device, temperature=1.0):
    import torch
    rates = board_to_rates(board, agent.num_neurons, agent.stim_rate)
    logits, _ = agent(rates.to(device))
    legal_moves = list(board.legal_moves)
    legal_idx = [move_to_index(m) for m in legal_moves]
    legal_logits = logits[0, legal_idx] / max(temperature, 1e-6)
    probs = torch.softmax(legal_logits, dim=-1)
    choice = torch.multinomial(probs, 1).item()
    return legal_moves[choice]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--opponent", default="stockfish", choices=["stockfish", "random", "human"])
    p.add_argument("--stockfish", default=None)
    p.add_argument("--agent-color", default="white", choices=["white", "black"])
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()

    agent = load_checkpoint(args.checkpoint, device=args.device)
    agent.eval()
    engine = make_engine(args.stockfish) if args.opponent == "stockfish" else None

    for g in range(args.games):
        board = chess.Board()
        agent_color = chess.WHITE if args.agent_color == "white" else chess.BLACK
        print(f"\n=== Game {g+1} ===")
        while not board.is_game_over():
            print(board)
            print()
            if board.turn == agent_color:
                move = choose_move(agent, board, args.device, args.temperature)
                print(f"Agent plays: {move.uci()}")
            else:
                if args.opponent == "stockfish" and engine is not None:
                    res = engine.play(board, chess.engine.Limit(depth=8))
                    move = res.move
                    print(f"Stockfish plays: {move.uci()}")
                elif args.opponent == "random":
                    move = random.choice(list(board.legal_moves))
                    print(f"Random plays: {move.uci()}")
                else:
                    move = chess.Move.from_uci(input("Your move (uci): ").strip())
            board.push(move)
        print(board)
        print("Result:", board.result())

    if engine is not None:
        engine.quit()


if __name__ == "__main__":
    main()
