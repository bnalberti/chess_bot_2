import chess
import chess.pgn
import random
from piece_matricies import *
from evaluate_position import *
from misc import *

def select_better_move(board, randomness=0.1, search_size=100):
    ###
    # Uses evaluates_position() to select the best move
    # Searches 100 legal moves by default to determine the best move
    # Occasionally still returns a random move
    ###

    current_player = board.turn

    # Sorting legal moves to find optimal captures first lowers runtime
    legal_moves = sorted(
        board.legal_moves,
        key=lambda m:mvv_lva(m, board),     # Creates a sort key based on MVV-LVA func
        reverse=True                        # Highest value returns first e.g. queen captured by pawn
    )

    # Searches the first n moves from the sorted legal moves defined above
    searched_moves = legal_moves[:search_size]

    # Occasionally return a random move for move diversity
    if random.random() < randomness:
        return random.choice(legal_moves)
    
    # Best moves for white are positive, negative for black
    # Positive and negative infinity bounds ensure it always finds a best move
    best_move = None
    best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

    # Choosing the actual move
    for move in searched_moves:
        board.push(move)                    # Temporarily pushes a move
        score = evaluate_position(board)    # Returns a score number

        # Setting best score and move for each player
        if current_player == chess.WHITE and score > best_score:
            best_score = score
            best_move = move
        elif current_player == chess.BLACK and score < best_score:
            best_score = score
            best_move = move

        board.pop()     # Reverts the board after temporarily checking the move

    return best_move or random.choice(legal_moves)