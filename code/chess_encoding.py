"""Encode a chess board as Poisson input rates for the fly brain."""

from __future__ import annotations

import chess
import torch

PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]
PIECE_TO_OFFSET = {p: i for i, p in enumerate(PIECE_ORDER)}

# first 64*12 neurons: square-major one-hot piece channels
BOARD_FEATURES = 64 * 12
SIDE_FEATURE = BOARD_FEATURES
CASTLE_FEATURES = SIDE_FEATURE + 1
EP_FEATURE = CASTLE_FEATURES + 4
TOTAL_BASE_FEATURES = EP_FEATURE + 1


def piece_channel(piece: chess.Piece) -> int:
    color_offset = 0 if piece.color == chess.WHITE else 6
    return color_offset + PIECE_TO_OFFSET[piece.piece_type]


def board_to_rates(board: chess.Board, num_neurons: int,
                   stim_rate: float = 200.0) -> torch.Tensor:
    """Return a (num_neurons,) float tensor of Poisson rates (Hz)."""
    rates = torch.zeros(num_neurons, dtype=torch.float32)
    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            idx = square * 12 + piece_channel(piece)
            if idx < num_neurons:
                rates[idx] = stim_rate

    # side to move: high rate when White to move
    if SIDE_FEATURE < num_neurons:
        rates[SIDE_FEATURE] = stim_rate if board.turn == chess.WHITE else 0.0

    # castling rights
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for i, ok in enumerate(rights):
        idx = CASTLE_FEATURES + i
        if idx < num_neurons and ok:
            rates[idx] = stim_rate

    # en passant target (single bit-ish rate)
    if board.ep_square is not None and EP_FEATURE < num_neurons:
        rates[EP_FEATURE] = stim_rate
    return rates
