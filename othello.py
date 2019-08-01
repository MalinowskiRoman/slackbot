'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
from othello_functions import *
from Agents import MLAgent, DenseBrain, HumanPlayer, Glutton, AlphaBeta
from Board import Board


brain = DenseBrain(128)
#brain.load_state_dict(torch.load('models/against_Glutton.pt'))

learning_AI = MLAgent(brain, optimizer=torch.optim.SGD(brain.parameters(), lr=0.01))
alpha_beta_AI = AlphaBeta(depth=3)
glutton_AI = Glutton()
glutton_AI2 = Glutton()
human = HumanPlayer()

# player can be from classes AlphaBeta, Glutton, HumanPlayer
def play_othello1(player1, player2, display=True):
    # white is -1
    # black is 1
    board = Board()
    player1.set_team('white')
    player2.set_team('black')
    team_val = -1
    while True:
        if display:
            print('')
            board.print()
        if not board.possible_moves(team_val):
            if not board.possible_moves(-team_val):
                break
            else:
                team_val = -team_val
                continue
        if team_val == -1:
            i, j = player1.play(board)
            if display: print(str(player1) + ': ' +chr(j + ord('a')) + str(i+1))
        else:
            i, j = player2.play(board)
            if display: print(str(player2) + ': ' + chr(j + ord('a')) + str(i+1))
        board.execute_turn((i,j), team_val)
        team_val = -team_val
        # print('////////////////////////////')
        # print_board1(definitive_coins(board))
        # print('////////////////////////////')
    if display: board.print()
    black, white = board.count_score()
    if display: print('Final score : \nWhite Team ({}) : {}\nBlack team ({}) : {}'.format(player1, white, player2, black))
    return white, black

nv = 0
i = 0
episode_length = 64
nb_ep = 0
while True:
    ep_victories = 0
    nb_ep += 1
    for ep in range(episode_length):
        i += 1
        white, black = play_othello1(learning_AI, glutton_AI, display=False)
        nv += (white - black > 0)
        ep_victories += (white - black > 0)
        learning_AI.next_game(white  -black)
        print('\r {} games played, LearningAI {} - {} Glutton'.format(i, nv, i - nv), end='')

    learning_AI.learn()
    print('\nEpisode {} finished, {} victories'.format(nb_ep, ep_victories))
    learning_AI.reset()
    torch.save(brain.state_dict(), 'models/against_Glutton.pt')

