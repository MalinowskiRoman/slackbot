import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from othello_functions import *
from Board import Board, Game
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
	def __init__(self, hidden_size, dropout=0):
		super().__init__()
		self.embeds = nn.Embedding(3, 3)
		self.layer1 = nn.Linear(3 * 64, hidden_size)
		self.drop = nn.Dropout(dropout)
		self.layer2 = nn.Linear(hidden_size, 64)

	def forward(self, board):
		embedded_board = self.embeds(board).view(-1)
		scores = self.layer2(F.relu(self.drop(self.layer1(embedded_board))))
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
	def __init__(self, brain, optimizer=None, team=None, name=None):
		super().__init__(team)
		self.brain = brain
		self.optimizer = optimizer
		self.move_history = [[]]
		self.reward_history = []
		self.name = 'LearningAI' if not name else name

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

	def train(self, adversaries, total_ep = 100, episode_length=64, path='models', save_name = 'against_glutton_and_self'):
		self.brain.train()
		n_ep = 0
		n_games = 0
		n_wins = 0
		wins = [0 for adv in adversaries]
		games = [0 for avd in adversaries]
		while n_ep < total_ep:
			ep_victories = 0
			n_ep += 1
			for ep in range(episode_length):
				n_games += 1
				k, adv = self.select_adversary(adversaries, wins, games, n_ep, path, save_name)
				print(
					'\rEpisode {}, game {} : playing against {}... '.format(n_ep, n_games, adv),
					end='')
				games[k] += 1
				game = Game(self, adv, display_func=None)
				black, white = game.rollout()
				win = (white - black > 0)
				wins[k] += win
				ep_victories += win
				n_wins += win
				self.next_game(white - black)
#				print('Win :)' if win else 'Lost :(')
			to_disp = []
			template = ''
			for i in range(len(wins)):
				to_disp.append(adversaries[i])
				to_disp.append(wins[i])
				to_disp.append(games[i])
				template += '{}: {}/{} | '
			template += 'Total : {totwin}/{total}'
			self.learn()
			print('\nEpisode {} finished, {} victories'.format(n_ep, ep_victories))
			print(template.format(*to_disp, totwin=n_wins, total=n_games))
			self.reset()
			torch.save(self.brain.state_dict(), os.path.join(path, '{}_{}.pt'.format(save_name, n_ep)))

	def select_adversary(self, adversaries, wins, games, nb_ep, path, save_name):
		distrib = np.array(
			[(games[i] - wins[i]) / (games[i] + 1) ** 2 + 1 / (2 ** i) for i in range(len(adversaries))])
		distrib /= np.sum(distrib)
		i = np.where(np.random.multinomial(1, distrib) == 1)[0][0]
		adv = adversaries[i]
		if adv == 'self':
			adv_brain = copy.deepcopy(self.brain)
			if nb_ep > 1:
				n_adv = np.random.randint(max(1, nb_ep - 4), nb_ep)
				adv_brain.load_state_dict(torch.load(os.path.join(path, '{}_{}.pt'.format(save_name, n_adv))))
			adv_brain.eval()
			adv = MLAgent(adv_brain)
		return i, adv


class Glutton(Player):
	def __init__(self, team=None):
		super().__init__(team)
		self.name = 'Glutton'

	def play(self, board):
		moves = board.possible_moves(self.team_val)
		score = []
		for move in moves:
			black, white = board.execute_turn(move, self.team_val, in_place=False).count_score()
			if self.team_val == 1:
				score.append(black)
			else:
				score.append(white)
		move = moves[score.index(max(score))]
		return move


class DiggingGlutton(Player):
	def __init__(self, team=None, depth=2):
		super().__init__(team)
		self.name = 'Glutton {}'.format(depth)
		self.depth=depth

	def play(self, board):
		move, _ = self.get_score(board, self.depth)
		return move

	def get_score(self, board, depth=2):
#		print('Depth : {}'.format(depth))
		best_score = -64
		my_moves = board.possible_moves(self.team_val)
#		print('My Moves : ' + str(my_moves))
		if not my_moves:
			return None, -64
		best_move = my_moves[0]
		for move in my_moves:
#			print('Testing move {} on board'.format(move))
#			board.print()
			dboard = board.execute_turn(move, self.team_val, in_place=False)
			if depth == 0:
				black, white = dboard.count_score()
				min_score = (black - white) * self.team_val
			else:
				min_score = 64
				adv_moves = dboard.possible_moves(-self.team_val)
#				print('His moves : {}'.format(adv_moves))
				for adv_move in adv_moves:
#					print('Assuming he plays {} on board'.format(adv_move))
#					dboard.print()
					ddboard = dboard.execute_turn(adv_move, -self.team_val, in_place=False)
					_, score = self.get_score(ddboard, depth=depth-1)
					min_score = min(min_score, score)
				if not adv_moves:
					_, min_score = self.get_score(dboard, depth=depth-1)
#			print('------------------------Min score for move {} is {}----------------------------'.format(move, min_score))
			if min_score >= best_score:
				best_move = move
				best_score = min_score
#		print('------------------------Max score is {}----------------------------'.format(best_score))
		return best_move, best_score


class AlphaBeta(Player):

	def __init__(self, depth, team=None, maximize=True):
		super().__init__(team)
		self.depth = depth
		self.turn_count = 0
		self.maximize = maximize
		self.name = 'AlphaBeta'

	def play(self, board):
		self.turn_count += 2
		return determine_alpha_beta(board, self.team_val, self.depth)

	def reset(self):
		self.turn_count = 0


class HumanPlayer(Player):
	def __init__(self, team=None):
		super().__init__(team)
		self.name = 'Human'

	def play(self, board):
		return determine_human(board, self.team_val)



