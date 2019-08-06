'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
import json
import os
from Agents import MLAgent, DenseBrain, HumanPlayer, Glutton, AlphaBeta, DiggingGlutton
from Board import Board, Game


class Arena:
	def __init__(self, nb_players, dir):
		self.players = []
		self.dir = dir
		self.nb_players = nb_players
		for i in range(nb_players):
			brain_size = np.random.randint(32, 129)
			dropout = 0.7 * np.random.rand()
			brain = DenseBrain(brain_size, dropout)
			lr = 0.05 * np.random.rand()
			momentum = np.random.rand()
			optim = torch.optim.SGD(brain.parameters(), lr=lr, momentum=momentum)
			agent = MLAgent(brain, optim)
			self.players.append(agent)
		self.records = [[(0,0) for i in range(nb_players)] for j in range(nb_players)]

	def wander(self, time = 1000000, learn_every=64, save_every=2):
		for episode in range(1, time):
			for i in range(self.nb_players):
				j = self.choose_fight(i)
				print('\rEpisode {}, game {}/{} : AI {} vs AI {}'.format(episode, i+1, self.nb_players, i, j), end='')
				game = Game(self.players[i], self.players[j], display_func=None)
				black, white = game.rollout()
				wins, games = self.records[i][j]
				self.players[i].next_game(white - black)
				self.players[j].next_game(black - white)
				self.records[i][j] = (wins + int(white - black > 0), games + 1)
				self.records[j][i] = (wins + int(black - white > 0), games + 1)
			if episode % learn_every == 0:
				for q, agent in enumerate(self.players):
					agent.learn()
					agent.reset()
				print('')
				self.print_charts()
				if (episode // learn_every) % save_every == 0:
					for i in range(self.nb_players):
						torch.save({'model': self.players[i].brain.state_dict(), 'opt': self.players[i].optimizer.state_dict()}, os.path.join(self.dir, 'agent_{}.pt'.format(i)))
					with open(os.path.join(self.dir, 'records.json'), 'w') as f:
						f.write(json.dumps(self.records))

	def choose_fight(self, i):
		record = self.records[i]
		distrib = np.array([1/(1 + record[j][0])**2 if j != i else 0 for j in range(self.nb_players)])
		distrib /= np.sum(distrib)
		return np.where(np.random.multinomial(1, distrib) == 1)[0][0]

	def print_charts(self):
		tracks = []
		for i in range(self.nb_players):
			ws, gs = 0, 0
			for j in range(self.nb_players):
				ws += self.records[i][j][0]
				gs += self.records[i][j][1]
			tracks.append((i, ws, gs))
		tracks.sort(key = lambda track: track[1])
		for rk, (i, wins, games) in enumerate(tracks[::-1]):
			agent = self.players[i]
			dim = agent.brain.layer1.out_features
			dropout = agent.brain.drop.p
			lr = agent.optimizer.defaults['lr']
			momentum = agent.optimizer.defaults['momentum']
			print('{:<2}. Agent {:<2} [Inner dim: {:<3}, dropout: {:<3.2f}, lr: {:<5.4f}, momentum: {:<3.2f}]: {:>4} wins ({} games)'.format(rk, i, dim, dropout, lr, momentum, wins, games))


arena = Arena(32, 'arena')
players = []
params = [{} for i in range(32)]
with open('arena/params.txt', 'r') as f:
	for line in f:
		i = int(line[10:12])
		dim = int(line[25:28])
		dropout = float(line[39:43])
		lr = float(line[49:55])
		momentum = float(line[67:71])
		params[i] = {'dim': dim, 'dropout': dropout, 'lr': lr, 'momentum': momentum}
for i in range(32):
	dict = torch.load('arena/agent_{}.pt'.format(i))
	brain = DenseBrain(params[i]['dim'], params[i]['dropout'])
	brain.load_state_dict(dict['model'])
	opt = torch.optim.SGD(brain.parameters(), lr=params[i]['lr'], momentum=params[i]['momentum'])
	agent = MLAgent(brain, opt)
	players.append(agent)
arena.players = players
with open('arena/records.json', 'r') as f:
	arena.records = json.load(f)
arena.wander()


def train(players, length = 10):
	# every player meet every other length times
	print('Train phase !')
	N = len(players)
	for k in range(length):
		for i, agent1 in enumerate(players):
			for j, agent2 in enumerate(players):
				if i != j:
					print('\rRound {}/{} - {:.2f}%'.format(k+1, length, 100*(N*i+j+1)/(N**2)), end='')
					game = Game(agent1, agent2, display_func=None)
					agent1.brain.train()
					agent2.brain.train()
					black, white = game.rollout()
					agent1.next_game(white - black)
					agent2.next_game(black - white)
		print('\rRound {}/{} - 100%'.format(k+1, length), end='')
		for agent in players:
			agent.learn()
			agent.reset()
	print('')


def eliminate(players, nb_games=32, ratio=0.3):
	print('Turnament phase !')
	for agent in players:
		agent.brain.eval()
		agent.nb_wins = 0
	N = len(players)
	for i, agent1 in enumerate(players):
		for j, agent2 in enumerate(players):
			if i != j:
				score = 0
				for k in range(nb_games):
					print('\r Game {}/{} - Round {}'.format(N*i+j+1, N*N, k), end='')
					game = Game(agent1, agent2, display_func=None)
					black, white = game.rollout()
					score += int((white > black)) - int((black > white))
				if score > 0:
					agent1.nb_wins += 1
				else:
					agent2.nb_wins += 1
	players.sort(key = lambda agent: agent.nb_wins)
	players = players[max(1, int(ratio*len(players))):]
	print('\n{} players remaining - {}'.format(len(players), [agent.name for agent in players]))
	return players


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
def tournament(players, nb_train_rounds = 16, nb_eval_rounds = 16):
	while len(players) > 1:
		train(players, nb_train_rounds)
		players = eliminate(players, nb_eval_rounds)
		winner = players[0]
		print('Done! Winner is {}'.format(winner.name))
		torch.save(winner.brain.state_dict(), 'models/winner_brain.pt')
		return winner


# play_othello1(alpha_beta_AI, alpha_beta_AI2)
def test_choices():
	board = Board()
	board.grid[2,1] = board.grid[2,2] = board.grid[2,3] = -1
	board.grid[3,4] = board.grid[4,3] = -1
	board.grid[3,3] = board.grid[4,4] = 1
	print(board)
	alpha_beta_AI2.set_team('black')
	tree = alpha_beta_AI2.play(board, test=True)
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
