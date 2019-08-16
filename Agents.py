import copy
import logging
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from Board import Board, Game

logging.basicConfig(encoding='utf-8', level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)

global corners
corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
global c_square
c_square = [(0, 1), (0, 7), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
global x_square
x_square = [(1, 1), (6, 6), (1, 6), (6, 1)]

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
        return F.softmax(scores, dim=0)

class StateActionPolicy(nn.Module):
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        self.embeds = nn.Embedding(3, 3)
        self.layer1 = nn.Linear(3 * 64, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.move_scores = nn.Linear(hidden_size, 64)
        self.state_score = nn.Linear(hidden_size, 1)

    def forward(self, board):
        embedded_board = self.embeds(board).view(-1)
        hidden = F.relu(self.drop(self.layer1(embedded_board)))
        scores = self.move_scores(hidden)
        value = self.state_score(hidden)
        return F.log_softmax(scores, dim=0), torch.tanh(value)


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
		valid = torch.zeros(64).scatter(0, torch.LongTensor([8*x+y for x, y in valid_moves]), 1).type(torch.ByteTensor)
		board = convert_board(board, self.team_val)
		scores = self.brain(board)
		scores = torch.where(valid, scores, torch.full((64,), -1000))
		sampler = torch.distributions.categorical.Categorical(logits=scores)
		move = sampler.sample()
		ctr = 0
		while not valid[move.item()]:
			if ctr > 5:
				logger.info(valid)
				logger.info(scores)
				logger.info([self.brain.parameters()])
			move = sampler.sample()
		self.move_history[-1].append(sampler.log_prob(move).unsqueeze(0))
		return move.item() // 8, move.item() % 8

    def compute_gradients(self):
        self.brain.zero_grad()
        self.reward_history = torch.FloatTensor(self.reward_history)
        self.reward_history -= self.reward_history.mean()
        self.reward_history /= (1 + self.reward_history.std(unbiased=False))
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

    def train(self, adversaries, total_ep=100, episode_length=64, path='models', save_name='against_glutton_and_self'):
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
                logger.info(
                    '\rEpisode {}, game {} : playing against {}... '.format(n_ep, n_games, adv),
                    end='')
                games[k] += 1
                if np.random.rand() > 0.5:
                    game = Game(self, adv, display_func=None)
                else:
                    game = Game(adv, self, display_func=None)
                black, white = game.rollout()
                win = (white - black > 0)
                wins[k] += win
                ep_victories += win
                n_wins += win
                self.next_game(white - black)
                if isinstance(adv, AlphaBeta):
                    adv.reset()
            #				logger.info('Win :)' if win else 'Lost :(')
            to_disp = []
            template = ''
            for i in range(len(wins)):
                to_disp.append(adversaries[i])
                to_disp.append(wins[i])
                to_disp.append(games[i])
                template += '{}: {}/{} | '
            template += 'Total : {totwin}/{total}'
            self.learn()
            logger.info('\nEpisode {} finished, {} victories'.format(n_ep, ep_victories))
            logger.info(template.format(*to_disp, totwin=n_wins, total=n_games))
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
        #		logger.info('Depth : {}'.format(depth))
        best_score = -64
        my_moves = board.possible_moves(self.team_val)
        #		logger.info('My Moves : ' + str(my_moves))
        if not my_moves:
            return None, -64
        best_move = my_moves[0]
        for move in my_moves:
            #			logger.info('Testing move {} on board'.format(move))
            dboard = board.execute_turn(move, self.team_val, in_place=False)
            if depth == 0:
                black, white = dboard.count_score()
                min_score = (black - white) * self.team_val
            else:
                min_score = 64
                adv_moves = dboard.possible_moves(-self.team_val)
                #				logger.info('His moves : {}'.format(adv_moves))
                for adv_move in adv_moves:
                    #					logger.info('Assuming he plays {} on board'.format(adv_move))
                    #					dboard.print()
                    ddboard = dboard.execute_turn(adv_move, -self.team_val, in_place=False)
                    _, score = self.get_score(ddboard, depth=depth - 1)
                    min_score = min(min_score, score)
                if not adv_moves:
                    _, min_score = self.get_score(dboard, depth=depth - 1)
            #			logger.info('------------------------Min score for move {} is {}----------------------------'.format(move, min_score))
            if min_score >= best_score:
                best_move = move
                best_score = min_score
        #		logger.info('------------------------Max score is {}----------------------------'.format(best_score))
        return best_move, best_score


class AlphaBeta(Player):
	def __init__(self, depth, team=None):
		super().__init__(team)
		self.depth = depth
		self.turn_count = 0
		self.name = 'AlphaBeta'

	def play(self, board, test=False):
		list_moves = board.possible_moves(self.team_val)
		for move in corners:
			if move in list_moves:
				return move
		if not test:
			score = []
			for move in list_moves:
				score += [self.alpha_beta(board.update(move, self.team_val, in_place=False), self.team_val, self.depth, alpha=-np.inf, beta=np.inf, maximize=False)]
			return list_moves[score.index(max(score))]
		else:
			tree = [board, 0, []]
			for move in list_moves:
				tree[2] += [self.alpha_beta_tree(board.update(move, self.team_val, in_place=False), self.team_val, self.depth, alpha=-np.inf, beta=np.inf, branch=[], maximize=False)]
			tree[1] = max([tree[2][i][1] for i in range(len(tree[2]))])
			return tree

	def alpha_beta(self, board, team_val, depth, alpha, beta, maximize=True):
		# We have to return the score of the root team (in the algorithm
		# the other team want to minimize this one, not maximizing it, even if
		# is equivalent in othello
		if depth == 0 or not board.possible_moves(team_val):
			# return evaluate_score(board_param, team_val, turn_count, maximize)
			return board.compute_score(team_val, maximize)
		else:
			team_val = - team_val
			if maximize:
				max_ = - np.inf
				for move in board.possible_moves(team_val):
					val = self.alpha_beta(board.update(move, team_val, in_place=False), team_val, depth - 1, alpha, beta, not maximize)
					max_ = max(max_, val)
					alpha = max(alpha, max_)
					if alpha >= beta:
						break
				return max_
			else:

				min_ = np.inf
				for move in board.possible_moves(team_val):
					val = self.alpha_beta(board.update(move, team_val, in_place=False), team_val, depth - 1, alpha, beta, not maximize)
					min_ = min(min_, val)
					beta = min(beta, min_)
					if alpha >= beta:
						break
				return min_

    def alpha_beta_tree(self, board, team_val, depth, alpha, beta, branch, maximize=True):
        # We have to return the score of the root team (in the algorithm
        # the other team want to minimize this one, not maximizing it, even if
        # is equivalent in othello
        if depth == 0 or not board.possible_moves(team_val):
            # return evaluate_score(board_param, team_val, turn_count, maximize)
            return board, board.compute_score(team_val, maximize)
        else:
            team_val = - team_val
            branch += [board, 0, []]
            if maximize:
                max_ = - np.inf
                for move in board.possible_moves(team_val):
                    val = self.alpha_beta_tree(board.update(move, team_val, in_place=False), team_val, depth - 1, alpha,
                                               beta, [], not maximize)
                    max_ = max(max_, val[1])
                    alpha = max(alpha, max_)
                    branch[2] += [val]
                    if alpha >= beta:
                        break
                branch[1] = max_
                return branch
            else:

                min_ = np.inf
                for move in board.possible_moves(team_val):
                    val = self.alpha_beta_tree(board.update(move, team_val, in_place=False), team_val, depth - 1, alpha,
                                               beta, [], not maximize)
                    min_ = min(min_, val[1])
                    beta = min(beta, min_)
                    branch[2] += [val]
                    if alpha >= beta:
                        break
                    branch[1] = min_
                return branch

    def reset(self):
        self.turn_count = 0


class HumanPlayer(Player):
    def __init__(self, team=None):
        super().__init__(team)
        self.name = 'Human'

    def play(self, board):
        if self.team_val == 1:
            team = 'black'
        else:
            team = 'white'
        format_ok = False
        while not format_ok:
            logger.info('\nTime for the ' + team + ' team to play !')
            txt = input('Where do you want to place a pawn ?\n\n')
            logger.info('')
            try:
                j = ord(txt[0].lower()) - ord('a')
                i = int(txt[1]) - 1
            except:
                logger.info('Use a format like "b7" !')
                continue
            if i in range(8) and j in range(8):
                if board.is_move_possible((i, j), self.team_val):
                    format_ok = True
                else:
                    logger.info("You can't put it there")
            else:
                logger.info('Use a format like "b7" !')
        return i, j


class MCTS(Player):
    def __init__(self, time):
        self.name = 'MCTS'
        self.time = time
        self.player1 = RandomPlayer()
        self.player2 = RandomPlayer()
        self.tree = GameNode(board=Board(), cur_team=-1)

    def expand_tree(self):
        node = self.tree
        node.visit_number += 1
        ctr = 0
        while not node.is_leaf:
            ctr += 1
            if ctr > 100:
                logger.info(node)
            node = node.choose_child()
        node.expand()
        if not node.is_terminal:
            try:
                node = node.choose_child(random=True)
            except ValueError as err:
                logger.info(node.children)
                raise err
        value = node.simulate_game(self.player1, self.player2)
        node.update(value)

    def play(self, board):
        self.update_root(board)
        start = time.time()
        while time.time() - start < self.time:
            self.expand_tree()
        best_move, node = max(self.tree.children.items(), key=lambda child: child[1].visit_number)
        try:
            assert board.is_move_possible(best_move, self.team_val)
        except AssertionError:
            logger.info(self.tree.children)
            logger.info(best_move)
            logger.info(board.possible_moves(self.team_val))
            logger.info(board.possible_moves(-self.team_val))
            raise AssertionError
        self.tree = node
        self.tree.parent = None
        return best_move

    def update_root(self, board):
        if not (self.tree.board == board and self.tree.cur_team == self.team_val):
            found = self.tree.look_for_board(board, self.team_val)
            if found:
                self.tree = found
                self.tree.parent = None
            else:
                self.tree = GameNode(board, cur_team=self.team_val)

    def set_team(self, team):
        super().set_team(team)
        self.tree.root_team = self.team_val


class GameNode:
    def __init__(self, board, parent=None, cur_team=None):
        self.parent = parent
        self.board = board
        self.cur_team = -parent.cur_team if parent else cur_team
        self.root_team = parent.root_team if parent else cur_team
        self.value = 0
        self.is_leaf = True
        self.is_terminal = False
        self.children = {}
        self.visit_number = 0

    def expand(self):
        assert self.is_leaf
        possible_moves = self.board.possible_moves(self.cur_team)
        if possible_moves:
            self.is_leaf = False
            self.children = {
                move: GameNode(parent=self, board=self.board.execute_turn(move, self.cur_team, in_place=False)) for move
                in possible_moves}
        else:
            if self.board.possible_moves(-self.cur_team):
                self.is_leaf = False
                node = GameNode(parent=self, board=copy.deepcopy(self.board))
                self.children = {None: node}
            else:
                self.is_terminal = True

    def update(self, value):
        self.value += value
        if self.parent:
            self.parent.update(value)

    def choose_child(self, random=False):
        if self.is_leaf:
            raise ValueError("Cannot choose a child from a leaf")
        nodes = list(self.children.values())
        values = np.array([self.cur_team * self.root_team * node.value / (1 + node.visit_number) for node in nodes])
        visits = np.array([node.visit_number for node in nodes])
        distrib = values + np.sqrt(2 * self.visit_number / (1 + visits))
        if not random:
            chosen = nodes[distrib.argmax()]
        else:
            chosen = np.random.choice(nodes)
        chosen.visit_number += 1
        return chosen

    def simulate_game(self, player1, player2):
        game = Game(player1, player2, display_func=None, board=copy.deepcopy(self.board), cur_team=self.cur_team)
        black, white = game.rollout()
        return int(self.root_team * (black - white) > 0)

    def look_for_board(self, board, team):
        for child in self.children.values():
            if child.board == board and child.cur_team == team:
                return child
        for child in self.children.values():
            found = child.look_for_board(board, team)
            if found:
                return found
        return None

    def c_str(self, depth=0):
        tree = '{}{} | {}/{}'.format('\t' * depth, self.cur_team, self.value, self.visit_number)
        for child in self.children.values():
            tree += '\n' + child.c_str(depth=depth + 1)
        return tree

    def __str__(self):
        return self.c_str()


class PolicyGameNode(GameNode):
    def __init__(self, board, parent=None, prob=1, cur_team=None):
        super().__init__(board, parent, cur_team)
        self.prob = prob

    def choose_child(self, random=False):
            if self.is_leaf:
                raise ValueError("Cannot choose a child from a leaf")
            nodes = list(self.children.values())
            values = np.array([self.cur_team * self.root_team * node.value / (1 + node.visit_number) for node in nodes])
            visits = np.array([node.prob*np.sqrt(2 * self.visit_number / (1 + node.visit_number)) for node in nodes])
            distrib = values + visits
            if not random:
                chosen = nodes[distrib.argmax()]
            else:
                chosen = np.random.choice(nodes)
            chosen.visit_number += 1
            return chosen

    def expand(self, network):
        assert self.is_leaf
        possible_moves = self.board.possible_moves(self.cur_team)
        if possible_moves:
            self.is_leaf = False
            log_probs, value = network(convert_board(self.board, self.cur_team))
            probs = torch.exp(log_probs)
            self.update(value.item()*self.cur_team*self.root_team)
            self.children = {
                (x,y): PolicyGameNode(parent=self, board=self.board.execute_turn((x,y), self.cur_team, in_place=False), prob=probs[8*x+y].item()) for x,y
                in possible_moves}
        else:
            if self.board.possible_moves(-self.cur_team):
                self.is_leaf = False
                node = PolicyGameNode(parent=self, board=copy.deepcopy(self.board), prob=1.)
                self.children = {None: node}
            else:
                self.is_terminal = True
                black, white = self.board.count_score()
                self.update(self.root_team * (2 * int(black - white > 0) - 1))

    def c_str(self, depth=0):
        tree = '{}{}| p = {:.3f} | value = {:.3f} | visit = {}'.format('\t' * depth, self.cur_team, self.prob, self.value, self.visit_number)
        for child in self.children.values():
            tree += '\n' + child.c_str(depth=depth + 1)
        return tree

    def __str__(self):
        return self.c_str()


class Alpha(Player):
    def __init__(self, network, time, optimizer, temperature=0.5):
        super().__init__()
        self.name = 'Alpha'
        self.time = time
        self.network = network
        self.tree = PolicyGameNode(board=Board(), cur_team=-1)
        self.temperature = temperature
        self.optimizer = optimizer
        self.move_history = []
        self.history = []


    def expand_tree(self):
        node = self.tree
        node.visit_number += 1
        ctr = 0
        while not node.is_leaf:
            ctr += 1
            if ctr > 100:
                logger.info(node)
            node = node.choose_child()
        node.expand(self.network)

    def play(self, board):
        self.update_root(board)
        start = time.time()
        nb_exp = 0
        self.network.eval()
        while time.time() - start < self.time:
            nb_exp += 1
            self.expand_tree()
        policy = torch.zeros(64)
        for (x,y), child in self.tree.children.items():
            policy[8*x+y] = np.power(child.visit_number, 1/self.temperature)
        policy /= torch.sum(policy)

        self.move_history.append([board, policy])
        sampler = torch.distributions.categorical.Categorical(probs=policy)
        move = sampler.sample().item()
        move = move // 8, move % 8
        try:
            assert board.is_move_possible(move, self.team_val)
        except AssertionError:
            logger.info(self.tree.children)
            logger.info(move)
            logger.info(board.possible_moves(self.team_val))
            logger.info(board.possible_moves(-self.team_val))
            raise AssertionError
        self.tree = self.tree.children[move]
        self.tree.parent = None
        return move

    def update_root(self, board):
        if not (self.tree.board == board and self.tree.cur_team == self.team_val):
            found = self.tree.look_for_board(board, self.team_val)
            if found:
                self.tree = found
                self.tree.parent = None
            else:
                self.tree = PolicyGameNode(board, cur_team=self.team_val)

    def set_team(self, team):
        super().set_team(team)
        self.tree.root_team = self.team_val

    def next_game(self, reward):
        for data in self.move_history:
            data.append(2 * int(reward > 0) - 1)
        self.history.extend(self.move_history)
        self.move_history = []

    def learn(self, sample_size=64):
        indexes = np.random.choice(len(self.history), min(sample_size, len(self.history)), replace=False)
        training_data = np.array(self.history)[indexes]
        self.network.train()
        for data in training_data:
            self.network.zero_grad()
            board, policy, value = tuple(data)
            policy_guess, value_guess = self.network(convert_board(board, self.team_val))
            loss = (value_guess - value)**2 - torch.sum(policy * policy_guess)
            loss.backward()
            self.optimizer.step()
        self.history = []

class RandomPlayer(Player):
    def __init__(self):
        self.name = 'RandomPlayer'

    def play(self, board):
        moves = board.possible_moves(self.team_val)
        return moves[np.random.choice(len(moves))]
