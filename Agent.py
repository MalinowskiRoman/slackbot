import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_board(board, team):
	"""
	convert the boards to contain 0,1,2 index : 0 is empty, 1 is current team and 2 is other team
	:param team: 'white' or 'black'
	:return: the converted board
	"""
	if team == 'white':
		idx = {-1: 1, 0: 0, 1:2}
	elif team == 'black':
		idx = {-1: 2, 0: 0, 1: 1}
	return [idx[board[i]] for i in range(64)]


class Brain(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.embeds = nn.Embedding(3, 3, padding_idx=0)
		self.layer1 = nn.Linear(3 * 64, hidden_size)
		self.layer2 = nn.Linear(hidden_size, 64)

	def forward(self, board):
		embedded_board = self.embeds(board).view(-1)
		scores = self.layer2(F.relu(self.layer1(embedded_board)))
		return F.softmax(scores, dim=0)


class Agent:
	def __init__(self, team, brain, optimizer):
		self.brain = brain
		self.optimizer = optimizer
		self.history = []
		self.team = team

	def play(self, board, valid_moves):
		flat_board = convert_board(torch.LongTensor(board).flatten(), self.team)
		distribution = self.brain(flat_board)
		flat_valid = torch.LongTensor([8 * x + y for x, y in valid_moves])
		valid_distribution = torch.zeros(64).scatter(0, flat_valid, distribution)
		move = torch.multinomial(valid_distribution, 1).item()
		self.history.append((distribution, move))
		return move // 8, move % 8

	def compute_gradients(self, score):
		for distribution, move in self.history:
			out_grad = torch.zeros(64)
			out_grad[move] = score
			distribution.backward(out_grad)

	def learn(self):
		self.optimizer.step()
		self.brain.zero_grad()

	def reset(self):
		self.history = []
		self.brain.zero_grad()
