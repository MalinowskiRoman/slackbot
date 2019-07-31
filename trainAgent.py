import torch
import othello
import numpy as np
from Agent import Agent, Brain


hidden_size = 128
nb_games = 1000
lr = 0.01
momentum = 0
save_path = 'models/'
brain = Brain(hidden_size)
opt = torch.optim.SGD(brain.parameters(), lr=lr, momentum=momentum)

agent_white = Agent('white', brain, opt)
agent_black = Agent('black', brain, opt)
white_victories = 0
for n in range(nb_games):
	agent_white.reset()
	agent_black.reset()

	board = [[0 for i in range(8)] for j in range(8)]
	board[3][3] = board[4][4] = 1
	board[4][3] = board[3][4] = -1
	team = 'white'
	team_val = -1
	while True:
		agent = agent_white if team == 'white' else agent_black
		valid_moves = othello.possible_moves(board, team_val)
		i, j = agent.play(board, valid_moves)
		next_move, board = othello.execute_turn(i, j, board, team_val)
		if next_move:
			if team == 'white':
				team = 'black'
			else:
				team = 'white'
			team_val = - team_val
			continue
		elif not othello.possible_moves(board, team_val):
			break
	black, white = othello.count_score(board)
	agent_white.compute_gradients(white - black)
	agent_black.compute_gradients(black - white)
	agent_white.learn()
	agent_black.learn()
	white_victories += (white > black)
	print('\r White {} - {} Black ! [{} games played, {} white victories, {} black victories]'.format(white, black, n + 1, white_victories, n - white_victories), end = '')
	torch.save(brain.state_dict(), save_path + 'game_{}.pt'.format(n))

