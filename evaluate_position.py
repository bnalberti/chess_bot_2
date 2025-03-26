import chess
import chess.pgn
from piece_matricies import *
from misc import *

def evaluate_position(board):
    ###
    # Calculates a score based on number of caputred pieces
    # and king safety
    ###

    score = 0

    # Rewards central squares
    central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for square in central_squares:
        if board.is_attacked_by(chess.WHITE, square):
            score += 0.5
        if board.is_attacked_by(chess.BLACK, square):
            score -= 0.5
    # Rewards king-distance in endgames
    if len(board.piece_map()) <= 10:
        black_king = board.king(chess.BLACK)
        white_king = board.king(chess.WHITE)
        score += 0.05 * chess.square_distance(white_king, black_king)

    # Penalizing doubled pawns
    doubled_pawns = 0
    for file in range(8):
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file])
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file])
        if white_pawns > 1:
            doubled_pawns += white_pawns - 1
        if black_pawns > 1:
            doubled_pawns -= black_pawns - 1

    # Rewarding passed pawns
    for pawn_square in board.pieces(chess.PAWN, chess.WHITE):
        # Get every black pawn that can attack this white pawn
        enemy_pawn_attack_mask = chess.BB_PAWN_ATTACKS[chess.BLACK][pawn_square]
        if not board.pieces(chess.PAWN, chess.BLACK) & enemy_pawn_attack_mask:
            # Score scales with advancement down board
            score += 0.5 * (6 - chess.square_rank(pawn_square))
    for pawn_square in board.pieces(chess.PAWN, chess.BLACK):
        # Opposite with black pawns
        enemy_pawn_attack_mask = chess.BB_PAWN_ATTACKS[chess.WHITE][pawn_square]
        if not board.pieces(chess.PAWN, chess.WHITE) & enemy_pawn_attack_mask:
            score -= 0.5 * (6 - chess.square_rank(pawn_square) - 1)

    # For every square with a piece on it
    for square, piece in board.piece_map().items():
        value = piece_values[piece.piece_type]              # Value per piece
        table = piece_heatmaps.get(piece.piece_type, None)  # Mapping piece heatmap to a variable

        row, col = divmod(square, 8)        # Dividing into an 8x8 chessboard

        if table:
            if piece.color == chess.BLACK:  # Flipping the rows for black pieces
                row = 7 - row

            value += table[row][col]           # Adding the heatmap values to value variable
        
        # White strives to maximize score, black strives to minimize score
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
        
        ### Evaluating hanging pieces ###
        white_attackers = board.attackers(chess.WHITE, square)
        black_attackers = board.attackers(chess.BLACK, square)

        if piece.color == chess.WHITE:
            defenders = white_attackers
            attackers = black_attackers
        else:
            defenders = black_attackers
            attackers = white_attackers

        # Penalize if piece is attacked and undefended
        if attackers and not defenders:
            penalty = piece_values[piece.piece_type] * (-20 if piece.color == chess.WHITE else 20)
            score += penalty
        
        # Reward if piece can be captured by a lower valued piece
        if attackers and defenders:
            weakest_attacker = min(piece_values[board.piece_at(sq).piece_type] for sq in attackers)
            weakest_defender = min(piece_values[board.piece_at(sq).piece_type] for sq in defenders)
            if weakest_attacker < weakest_defender:
                trade_bonus = (weakest_defender - weakest_attacker)
                score += trade_bonus * (-3 if piece.color == chess.WHITE else 3)

    # If king in check, remove points from score
    if board.is_check():
        if board.turn == chess.WHITE:
            score -= 2
        else:
            score += 2
    
    # Small bonus for reserving castling rights
    if board.has_castling_rights(chess.WHITE):
        score += 0.5
    if board.has_castling_rights(chess.BLACK):
        score -= 0.5
    
    if has_castled(board, chess.WHITE):
        score += 3
    if has_castled(board, chess.BLACK):
        score -= 2
    
    return score