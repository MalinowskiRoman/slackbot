'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
from othello_functions import *
from Agents import MLAgent, DenseBrain, HumanPlayer, Glutton, AlphaBeta, DiggingGlutton
from Board import Board, Game

hidden_size = 128

brain = DenseBrain(hidden_size, dropout=0.4)
brain.load_state_dict(torch.load('models/best.pt'))

learning_AI = MLAgent(brain, optimizer=torch.optim.SGD(brain.parameters(), lr=0.01))
alpha_beta_AI = AlphaBeta(depth=2)
alpha_beta_AI2 = AlphaBeta(depth=2)
glutton_AI = Glutton()
glutton_AI2 = Glutton()
human = HumanPlayer()
dig_glutton = DiggingGlutton(depth=2)
randomAI = MLAgent(DenseBrain(32), name='randomAI')

adversaries = ['self', randomAI, DiggingGlutton(depth=0)]

learning_AI.train(adversaries, 10)

# brain.eval()
# game = Game(learning_AI, DiggingGlutton(depth=1))
# game.rollout()