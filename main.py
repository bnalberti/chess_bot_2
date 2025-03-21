import pandas as pd
import numpy as np
import chess
import chess.pgn
import random

##########
# Features for feature optimization
##########
# - Piece placement in a binary array (board_to_num func)
# - Material imbalance
# - King safety e.g. distance from enemy pieces
##########

def play_game(i):
    ###
    # Plays i games and prints out the result, move count, and ending positions board
    # Moves are decided randomly based on a list of legal moves
    # Next steps: connect winning games to a dataframe and append only winners, penalize losses and draws
    ###
    print('Playing games...')
    for _ in range(i):
        board = chess.Board()           # Initializes new chess board each loop
        move_num = 0
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)           # Make a random move from a list of legal moves
            board.push(move)                            # Push the random move
            move_num += 1
            result = board.result()
        print(f'Game over! Result: {result}')
        print(f'Total moves: {move_num}')
        print(board)

play_game(5)

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