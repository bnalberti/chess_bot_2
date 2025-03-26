import pandas as pd
import numpy as np
import chess
import chess.pgn
import os
from piece_matricies import *
from evaluate_position import *
from select_move import *
from misc import *

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
games_v3 = 'data/games_v3.csv'

def play_game(i, version_num):
    ###
    # Plays i games and prints out decisive games (final position) and imports them to a csv file
    # Moves are decided randomly based on a list of legal moves
    # Next steps: Start creating features to optimize moves slowly
    ###

    print('Playing games...')
    games_data = []                     # Decisive game results stored in a list

    df = load_existing_data(version_num)           # Load existing game data or creates new file

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

play_game(100, games_v3)



# Next steps: Reward  tempo (developing early pieces), pawn cover near king
# Create variables for all weights for ML optimization
# Modularize the program
# Currently the draw rate is ~80%, with ~20% ending in w/l in 30-70 moves
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