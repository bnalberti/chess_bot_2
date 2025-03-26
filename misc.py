import pandas as pd
import numpy as np
import chess
import chess.pgn
import os
from piece_matricies import *

piece_heatmaps = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}

piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

def load_existing_data(version_num):
    # If no persistent storage csv file, create one
    if os.path.exists(version_num) and os.path.getsize(version_num) > 0:
        return pd.read_csv(version_num)
    else:
        return pd.DataFrame(columns=['Winner', 'Move Count', 'Moves', 'Final Position'])

def has_castled(board, color):
    king_square = board.king(color)
    if color == chess.WHITE:
        # True if white castled king or queen side
        return king_square in [chess.G1, chess.C1]
    else:
        # True if black castled king or queen side
        return king_square in [chess.G8, chess.C8]
    
def mvv_lva(move, board):
        ###
        # Returns a number for how valuable any given capture is
        # E.g. queen(9) captured by pawn (1) is 9 - 1 = 8, knight(3) captured by rook(5) is 3 - 5 = -2
        # This ensures any valuable captures will be played first
        ###
        if not board.is_capture(move):
            return 0
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if not victim or not attacker:
            return 0
        return piece_values[victim.piece_type] - piece_values[attacker.piece_type]