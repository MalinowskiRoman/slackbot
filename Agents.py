import torch
import torch.nn as nn
import torch.nn.functional as F
from othello_functions import *

def convert_board(board, team):
	"""
	convert the boards to contain 0,1,2 index : 0 is empty, 1 is current team and 2 is other team
	:param team: -1 (white) or 1 (black)
	:return: the converted board
	"""
	if team == -1:
		idx = {-1: 1, 0: 0, 1:2}
	elif team == 1:
		idx = {-1: 2, 0: 0, 1: 1}
	return torch.LongTensor([[idx[board[i][j]] for i in range(8)] for j in range(8)])


class DenseBrain(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.embeds = nn.Embedding(3, 3)
		self.layer1 = nn.Linear(3 * 64, hidden_size)
		self.layer2 = nn.Linear(hidden_size, 64)

	def forward(self, board):
		embedded_board = self.embeds(board).view(-1)
		scores = self.layer2(F.relu(self.layer1(embedded_board)))
		return F.log_softmax(scores, dim=0)

# class ConvBrain(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.

class Player:
	def __init__(self, team=None):
		self.team_val = -1 if team =='white' else 1 if team == 'black' else None
		self.team = team
		self.name = 'Player'

	def play(self, board):
		pass

	def __str__(self):
		return self.name

	def set_team(self, team):
		assert team in ['white', 'black']
		self.team = team
		self.team_val = -1 if team == 'white' else 1


class MLAgent(Player):
	def __init__(self, brain, optimizer, team=None):
		super().__init__(team)
		self.brain = brain
		self.optimizer = optimizer
		self.move_history = [[]]
		self.reward_history = []
		self.name = 'LearningAI'

	def play(self, board):
		valid_moves = board.possible_moves(self.team_val)
		valid = torch.zeros(64).scatter(0, torch.LongTensor([8*x+y for x,y in valid_moves]), 1).type(torch.ByteTensor)
		board = convert_board(board, self.team_val)
		scores = self.brain(board)
		scores = torch.where(valid, scores, torch.full((64,), -1000))
		sampler = torch.distributions.categorical.Categorical(logits=scores)
		move = sampler.sample()
		self.move_history[-1].append(sampler.log_prob(move).unsqueeze(0))
		return move.item() // 8, move.item() % 8

	def compute_gradients(self):
		self.brain.zero_grad()
		self.reward_history = torch.FloatTensor(self.reward_history)
		self.reward_history -= self.reward_history.mean()
		self.reward_history /= self.reward_history.std()
		loss = torch.Tensor([0.])
		for i in range(len(self.reward_history)):
			moves = torch.cat(tuple(self.move_history[i]), dim=0)
			loss -= torch.sum(moves) * self.reward_history[i]
		loss /= len(self.reward_history)
		loss.backward()

	def learn(self):
		self.compute_gradients()
		torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 10)
		self.optimizer.step()

	def next_game(self, score):
		self.reward_history.append(score)
		self.move_history.append([])

	def reset(self):
		self.move_history = [[]]
		self.reward_history = []
		self.brain.zero_grad()


class Glutton(Player):
	def __init__(self, team=None):
		super().__init__(team)
		self.name = 'Glutton'

	def play(self, board):
		return determine_glutton(board, self.team_val)


class AlphaBeta(Player):
	def __init__(self, depth, team = None, corner_value=10, x_c_value=10, possibility_value=1, maximize=True):
		super().__init__(team)
		self.depth = depth
		self.turn_count = 0
		self.corner_value=corner_value
		self.x_c_value = x_c_value
		self.possibility_value = possibility_value
		self.maximize = maximize
		self.name = 'AlphaBeta'

	def play(self, board):
		self.turn_count += 2
		return determine_alpha_beta(board, self.team_val, self.turn_count, self.corner_value, self.x_c_value, self.possibility_value, self.depth)

	def reset(self):
		self.turn_count = 0


class HumanPlayer(Player):
	def __init__(self, team=None):
		super().__init__(team)
		self.name = 'Human'

	def play(self, board):
		return determine_human(board, self.team_val)