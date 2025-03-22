import pandas as pd
import numpy as np
import chess
import chess.pgn
import random
import os

##########
# Features for feature optimization
##########
# - Piece placement in a binary array (board_to_num func)
# - Material imbalance
# - King safety e.g. distance from enemy pieces
# - Number of moves
##########

games_v1 = 'data/games_v1.csv'

def load_existing_data():
    if os.path.exists(games_v1) and os.path.getsize(games_v1) > 0:
        return pd.read_csv(games_v1)
    else:
        return pd.DataFrame(columns=['Winner', 'Move Count', 'Moves', 'Final Position'])

def play_game(i):
    ###
    # Plays i games and prints out the result, move count, and ending positions board
    # Moves are decided randomly based on a list of legal moves
    # Next steps: Import the dataframe into a csv to begin storing decisive wins
    ###

    print('Playing games...')
    games_data = []                     # Decisive game results stored in a list

    for _ in range(i):
        board = chess.Board()           # Initializes new chess board each loop
        move_num = 0
        move_list = []

        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)           # Make a random move from a list of legal moves
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
    df = load_existing_data()
    new_df = pd.DataFrame(games_data)
    
    # Appends new data to the existing dataframe
    df = pd.concat([df, new_df], ignore_index=True)
    print(df)

    df.to_csv(games_v1, index=False)
    print(f'Saved games to persistent storage at {games_v1}')

play_game(5000)



# Next steps: record all games that do not draw into a dataframe
# Use feature optimization to extract datapoints from the winning side
# Currently the draw rate is ~90%, with ~10% ending in w/l in 200-600 moves
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