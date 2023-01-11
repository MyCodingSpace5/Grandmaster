# Overlord Chess Engine
import keras as k
import numpy as np
import chess.pgn
model = k.Sequential()
model.add(k.Dense(128, input_shape=(k.input_dim,), activation='relu'))
model.add(k.Dense(64, activation='relu'))
model.add(k.Dense(32, activation='relu'))
model.add(k.Dense(16, activation='relu'))
model.add(k.Dense(8, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
n_pieces = 3
n_squares = 8*8


board = np.array([[0,1,0,0,0,0,2,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0]
                 ])

one_hot_vector = np.zeros((n_pieces, n_squares))

for i in range(8):
    for j in range(8):
        if board[i][j] != 0:
            one_hot_vector[board[i][j]-1][i*8+j] = 1
pgn = open("master_games.pgn")

board_positions = []
moves = []

while 1:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    board = game.board()
    for move in game.main_line():
        board.push(move)
        # Append the board position and move to the lists
        board_positions.append(board.fen())
        moves.append(move)

model.fit(one_hot_vector,move,batch_size=12,epochs=100)
model.summary()
