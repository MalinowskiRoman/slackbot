'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
from othello_functions import *
from Agents import MLAgent, Brain, HumanPlayer, Glutton, AlphaBeta


brain = Brain(128)
brain.load_state_dict(torch.load('models/against_Glutton.pt'))

learning_AI = MLAgent('white', brain, optimizer=torch.optim.SGD(brain.parameters(), lr=0.01, momentum=0.9))
alpha_beta_AI = AlphaBeta('black', depth=0)
glutton_AI = Glutton('white')
glutton_AI2 = Glutton('black')


# player can be from classes AlphaBeta, Glutton, HumanPlayer
def play_othello1(player1=learning_AI, player2=alpha_beta_AI, display=True):
    # white is -1
    # black is 1

    board = [[0 for i in range(8)] for j in range(8)]
    board[3][3] = board[4][4] = 1
    board[4][3] = board[3][4] = -1

    team_val = -1

    while True:
        if display:
            print('')
            print_board1(board)



        if team_val == -1:
            i, j = player1.play(board)
            if display: print(str(player1) + ': ' + chr(j + ord('a')) + str(i))

        else:
            i, j = player2.play(board)
            if display: print(str(player2) + ': ' + chr(j + ord('a')) + str(i))

        next_move, board = execute_turn(i, j, board, team_val)

        if next_move:
            team_val = - team_val
        elif not possible_moves(board, team_val):
            break

    if display: print_board1(board)

    black, white = count_score(board)
    if isinstance(player1, MLAgent):
        player1.compute_gradients(white - black)
        player1.learn()
        player1.reset()
        torch.save(player1.brain.state_dict(), 'models/against_{}.pt'.format(player2))
    if isinstance(player2, MLAgent):
        player2.compute_gradients(white - black)
        player2.learn()
        player2.reset()
        torch.save(player2.brain.state_dict(), 'models/against_{}.pt'.format(player1))

    print('Final score : \nWhite Team ({}) : {}\nBlack team ({}) : {}'.format(player1, white, player2, black))


# play_othello1()
board = [[0 for i in range(8)] for j in range(8)]
board[3][3] = board[4][4] = -1
board[4][3] = board[3][4] = 1
board[2][1] = board[2][3] = 1
board[2][2] = 1
print_board1(board)
best, moves, score, values = determine_alpha_beta(board, -1, 5, 2)
print(best)
for i in range(len(moves)):
    print(str(moves[i]) + ', ' + str(score[i]) + ', ' + str(values[i]))
    print(2*values[i][1]+values[i][0]+1000*values[i][2]+500*values[i][3]-500*values[i][4])