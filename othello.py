'''Created by Roman Malinowski '''
import copy
import numpy as np
import torch
import json
import os
import time
from Agents import MLAgent, DenseBrain, HumanPlayer, Glutton, AlphaBeta, DiggingGlutton, MCTS, StateActionPolicy, Alpha, RandomPlayer
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



# arena = Arena(32, 'arena')
# players = []
# params = [{} for i in range(32)]
# with open('arena/params.txt', 'r') as f:
# 	for line in f:
# 		i = int(line[10:12])
# 		dim = int(line[25:28])
# 		dropout = float(line[39:43])
# 		lr = float(line[49:55])
# 		momentum = float(line[67:71])
# 		params[i] = {'dim': dim, 'dropout': dropout, 'lr': lr, 'momentum': momentum}
# for i in range(32):
# 	dict = torch.load('arena/agent_{}.pt'.format(i))
# 	brain = DenseBrain(params[i]['dim'], params[i]['dropout'])
# 	brain.load_state_dict(dict['model'])
# 	opt = torch.optim.SGD(brain.parameters(), lr=params[i]['lr'], momentum=params[i]['momentum'])
# 	agent = MLAgent(brain, opt)
# 	players.append(agent)
# arena.players = players
# with open('arena/records.json', 'r') as f:
# 	arena.records = json.load(f)
# arena.wander()

alpha_beta_AI = AlphaBeta(depth=2)
explorer = MCTS(time = 5.)
network = StateActionPolicy(hidden_size=128, dropout=0.3)
network.load_state_dict(torch.load('alpha.pt'))
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.8)
alpha = Alpha(network, time=0.2, optimizer=optimizer, temperature=0.5)
alpha_2 = Alpha(network, time=0.2, optimizer=optimizer)
random = RandomPlayer()

for k in range(100):
	for i in range(32):
		print('\rEpisode {} - Game {}'.format(k+1, i+1), end = '')
		game = Game(alpha, alpha_2, display_func=None)
		black, white = game.rollout()
		alpha.next_game(white - black)
		alpha_2.next_game(black - white)
	alpha.learn()
	alpha_2.learn()
	torch.save(network.state_dict(), 'alpha.pt')

