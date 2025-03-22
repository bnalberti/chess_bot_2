import pandas as pd
import numpy as np
import chess
import chess.pgn
import random
import os
from piece_matricies import *

##########
# Features for feature optimization
##########
# - Piece placement in a binary array (board_to_num func)
# - Material imbalance
# - King safety e.g. distance from enemy pieces
# - Number of moves
##########

# Each version of the bot will have ~1000 games for ML model to train off of
games_v1 = 'data/games_v1.csv'
games_v2 = 'data/games_v2.csv'

piece_heatmaps = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}

def load_existing_data(version_num):
    # If no persistent storage csv file, create one
    if os.path.exists(version_num) and os.path.getsize(version_num) > 0:
        return pd.read_csv(version_num)
    else:
        return pd.DataFrame(columns=['Winner', 'Move Count', 'Moves', 'Final Position'])

def play_game(i, version_num):
    ###
    # Plays i games and prints out decisive games (final position) and imports them to a csv file
    # Moves are decided randomly based on a list of legal moves
    # Next steps: Start creating features to optimize moves slowly
    ###

    print('Playing games...')
    games_data = []                     # Decisive game results stored in a list

    df = load_existing_data(games_v2)           # Load existing game data or creates new file

    for _ in range(i):
        board = chess.Board()           # Initializes new chess board each loop
        move_num = 0
        move_list = []

        while not board.is_game_over():
            move = select_better_move(board)            # Make a random move from a list of legal moves
            move_list.append(board.san(move))           # Stores move in SAN format
            board.push(move)                            # Push the random move
            move_num += 1
            
        result = board.result()
        if result == '1-0':
            winner = 'White'
        elif result == '0-1':
            winner = 'Black'
        else:
            continue                            # Skips draws

        # Append games to list
        games_data.append({
            'Winner': winner,
            'Move Count': move_num,
            'Moves': ' '.join(move_list),       # Converts list to a single string
            'Final Position': board.fen()       # Stores final board position as FEN
        })

        print(f'Game over! Result: {result}')
        print(f'Total moves: {move_num}')
        print(board)

    print('Transferring decisive games to dataframe below...')
    # Converts list of game data to dataframe
    new_df = pd.DataFrame(games_data)
    
    # Appends new data to the existing dataframe
    if not new_df.empty:
        df = pd.concat([df, new_df], ignore_index=True)
    
    # Ensures /data directory exists
    os.makedirs(os.path.dirname(version_num), exist_ok=True)
    print(df)

    df.to_csv(version_num, index=False)
    print(f'Saved {len(new_df)} new games to persistent storage at {version_num}')

def evaluate_position(board):
    ###
    # Calculates a score based on number of caputred pieces
    # and king safety
    ###

    score = 0

    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

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

    # If king in check, remove points from score
    if board.is_check():
        if board.turn == chess.WHITE:
            score -= 10
        else:
            score += 10
    
    
    return score

def select_better_move(board, randomness=0.1, search_size=100):
    ###
    # Uses evaluates_position() to select the best move
    # Searches 100 legal moves by default to determine the best move
    # Occasionally still returns a random move
    ###

    current_player = board.turn
    legal_moves = list(board.legal_moves)

    # Checking if the total num of legal moves is less than default search size
    if len(legal_moves) > search_size:
        searched_moves = random.sample(legal_moves, search_size)
    else:
        searched_moves = legal_moves

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

        board.pop()             # Reverts the board after temporarily checking the move

    return best_move or random.choice(legal_moves)

play_game(100, games_v2)



# Next steps: Setup ML to begin optimization
# Use feature optimization to extract datapoints from the winning side
# Currently the draw rate is ~85%, with ~15% ending in w/l in 30-70 moves
# Random moves cause very long games that end in stalemates

#--------------------------------------------------------------------#
def board_to_num(board):
    ###
    # Transforms a chess board into a numerical form for ML
    ###
    board_array = np.zeros((8, 8, 12), dtype=int)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,     # White pieces uppercase
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11    # Black pieces lowercase
    }
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)                            # Converts a 1D index into 2D board-coords
        row = 7 - row
        board_array[row][col][piece_map[piece.symbol()]] = 1    # Sets value to 1 for a piece in a given square
    return board_array.flatten()