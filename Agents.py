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
	return torch.LongTensor([idx[board[i].item()] for i in range(64)])


class Brain(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.embeds = nn.Embedding(3, 3)
		self.layer1 = nn.Linear(3 * 64, hidden_size)
		self.layer2 = nn.Linear(hidden_size, 64)

	def forward(self, board):
		embedded_board = self.embeds(board).view(-1)
		scores = self.layer2(F.relu(self.layer1(embedded_board)))
		return scores

class Player:
	def __init__(self, team):
		self.team_val = -1 if team =='white' else 1
		self.team = team
		self.name = 'Player'

	def play(self, board):
		pass

	def __str__(self):
		return self.name


class MLAgent(Player):
	def __init__(self, team, brain, optimizer):
		super().__init__(team)
		self.brain = brain
		self.optimizer = optimizer
		self.history = []
		self.name = 'LearningAI'

	def play(self, board):
		valid_moves = possible_moves(board, self.team_val)
		if not valid_moves:
			print('No moves')
			return 0, 0
		flat_board = convert_board(torch.LongTensor(board).flatten(), self.team_val)
		scores = self.brain(flat_board)
		flat_valid = torch.LongTensor([8 * x + y for x, y in valid_moves])
		valid_scores = ((-10000)*torch.ones(64)).scatter(0, flat_valid, scores)
		try:
			move = torch.multinomial(F.softmax(valid_scores, dim=0), 1).item()
		except RuntimeError as er:
			print_board1(board)
			print(scores)
			print(valid_scores)
			raise er
		self.history.append((scores, move))
		return move // 8, move % 8

	def compute_gradients(self, score):
		for distribution, move in self.history:
			out_grad = torch.zeros(64)
			out_grad[move] = score
			distribution.backward(out_grad/len(self.history))

	def learn(self):
		torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 10)
		self.optimizer.step()
		self.brain.zero_grad()

	def reset(self):
		self.history = []
		self.brain.zero_grad()


class Glutton(Player):
	def __init__(self, team):
		super().__init__(team)
		self.name = 'Glutton'

	def play(self, board):
		return determine_glutton(board, self.team_val)


class AlphaBeta(Player):
	def __init__(self, team, depth, corner_value=10, x_c_value=10, possibility_value=1, maximize=True):
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
	def __init__(self, team):
		super().__init__(team)
		self.name = 'Human'

	def play(self, board):
		return determine_human(board, self.team_val)