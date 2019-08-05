'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
from Agents import MLAgent, DenseBrain, HumanPlayer, Glutton, AlphaBeta, DiggingGlutton
from Board import Board, Game

hidden_size = 128

brain = DenseBrain(hidden_size, dropout=0.4)
brain.load_state_dict(torch.load('models/against_glutton_and_self_125.pt'))

learning_AI = MLAgent(brain, optimizer=torch.optim.SGD(brain.parameters(), lr=0.01))

alpha_beta_AI = AlphaBeta(depth=2)
alpha_beta_AI2 = AlphaBeta(depth=2)
glutton_AI = Glutton()
glutton_AI2 = Glutton()
human = HumanPlayer()
dig_glutton = DiggingGlutton(depth=2)

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
            print(board)
        if not board.possible_moves(team_val):
            if not board.possible_moves(-team_val):
                break
            else:
                team_val = -team_val
                continue
        if team_val == -1:
            i, j = player1.play(board)
            if display: print(str(player1) + ': ' + chr(j + ord('a')) + str(i+1))
        else:
            i, j = player2.play(board)
            if display: print(str(player2) + ': ' + chr(j + ord('a')) + str(i+1))
        board.execute_turn((i,j), team_val)
        team_val = -team_val
    if display: print(board)
    # print('////////////////////////////')
    # board.definitive_coins().print()
    # print('////////////////////////////')
    black, white = board.count_score()
    if display: print('Final score : \nWhite Team ({}) : {}\nBlack team ({}) : {}'.format(player1, white, player2, black))
    return white, black


# adversaries = ['self', DiggingGlutton(depth=0), DiggingGlutton(depth=1)]
# #brain.train()
# #learning_AI.train(adversaries, 10)
#
# brain.eval()
# game = Game(learning_AI, alpha_beta_AI)
# game.rollout()

# play_othello1(alpha_beta_AI, alpha_beta_AI2)
def test_choices():
    board = Board()
    board.grid[2,1] = board.grid[2,2] = board.grid[2,3] = -1
    board.grid[3,4] = board.grid[4,3] = -1
    board.grid[3,3] = board.grid[4,4] = 1
    print(board)
    alpha_beta_AI2.set_team('black')
    tree = alpha_beta_AI2.play(board,test=True)
    print('Begin Branch 0')
    for i in tree[2]:
        print('Begin Branch 1')
        for j in i[2]:
            print('Begin Branch 2')
            for k in j[2]:
                print(k[0])
                print(k[1])
                print('')
            print('end_branch 2')
            print('')
            print(j[0])
            print(j[1])
            print('')
        print('end_branch 1')
        print('')
        print(i[0])
        print(i[1])
        print('')
    print('end_branch 0')
    print(tree[0])
    print([tree[1]])
    print('\n')

test_choices()