'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
from othello_functions import *
from Agents import MLAgent, DenseBrain, HumanPlayer, Glutton, AlphaBeta, DiggingGlutton
from Board import Board

hidden_size = 128

brain = DenseBrain(hidden_size, dropout=0.4)
brain.load_state_dict(torch.load('models/against_glutton_and_self_17.pt'))

learning_AI = MLAgent(brain, optimizer=torch.optim.SGD(brain.parameters(), lr=0.01))
alpha_beta_AI = AlphaBeta(depth=3)
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
n = 0
episode_length = 64
nb_ep = 0
brain.eval()

adversaries = ['self'] + [DiggingGlutton(depth=k) for k in range(3)]
wins = [0 for adv in adversaries]
games = [0 for avd in adversaries]

def select_adversary(adversaries, wins, games, nb_ep):
    distrib = np.array([(games[i] - wins[i])/(games[i]+1)**2 + 1/(2**i) for i in range(len(adversaries))])
    distrib /= np.sum(distrib)
    i = np.where(np.random.multinomial(1, distrib) == 1)[0][0]
    adv = adversaries[i]
    if adv == 'self':
        adv_brain = DenseBrain(hidden_size, dropout=0.4)
        if nb_ep > 1:
            n_adv = np.random.randint(max(1, nb_ep - 4), nb_ep)
            adv_brain.load_state_dict(torch.load('models/against_glutton_and_self_{}.pt'.format(n_adv)))
        adv_brain.eval()
        adv = MLAgent(adv_brain)
    return i, adv


while True:
    ep_victories = 0
    nb_ep += 1
    for ep in range(episode_length):
        n += 1
        k, adv = select_adversary(adversaries, wins, games, nb_ep)
        print('Episode {}, game {} : playing against {} {}... '.format(nb_ep, n, adv, (k-1) if k != 0 else ''), end = '')
        games[k] += 1
        white, black = play_othello1(learning_AI, adv, display=False)
        win = (white - black > 0)
        nv += win
        wins[k] += win
        ep_victories += win
        learning_AI.next_game(white - black)
        print('Win :)' if win else 'Lost :(')
        to_disp = []
        for i in range(len(wins)):
            to_disp.append(wins[i])
            to_disp.append(games[i])

    learning_AI.learn()
    print('\nEpisode {} finished, {} victories'.format(nb_ep, ep_victories))
    print('self : {}/{}, Glutton 0: {}/{}, Glutton 1: {}/{}, Glutton 2: {}/{} --- {totwin}/{total} total'.format(*to_disp,totwin=nv,total=n))
    learning_AI.reset()
    torch.save(brain.state_dict(), 'models/against_glutton_and_self_{}.pt'.format(nb_ep))

brain.eval()
play_othello1(learning_AI, dig_glutton)