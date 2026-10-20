"""Fixed index mapping for chess moves."""

from __future__ import annotations

import chess

PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def generate_move_list() -> list[tuple[int, int, int | None]]:
    moves: list[tuple[int, int, int | None]] = []
    seen: set[tuple[int, int, int | None]] = set()
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            # normal move
            key = (from_sq, to_sq, None)
            if key not in seen:
                seen.add(key)
                moves.append(key)
            # promotions only when pawn moves to last rank
            from_rank = from_sq // 8
            to_rank = to_sq // 8
            if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                for promo in PROMOTION_PIECES:
                    key = (from_sq, to_sq, promo)
                    if key not in seen:
                        seen.add(key)
                        moves.append(key)
    return moves


MOVE_LIST = generate_move_list()
MOVE_TO_INDEX = {m: i for i, m in enumerate(MOVE_LIST)}
NUM_MOVES = len(MOVE_LIST)


def move_to_index(move: chess.Move) -> int:
    key = (move.from_square, move.to_square, move.promotion)
    if key not in MOVE_TO_INDEX:
        # fallback: register on the fly (should not happen for standard moves)
        MOVE_TO_INDEX[key] = len(MOVE_LIST)
        MOVE_LIST.append(key)
    return MOVE_TO_INDEX[key]


def index_to_move(idx: int) -> tuple[int, int, int | None]:
    return MOVE_LIST[idx]
