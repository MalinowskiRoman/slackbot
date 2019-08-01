import numpy as np
import copy

class Board:
	directions = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

	def __init__(self, grid=None):
		self.grid = np.zeros((8,8)) if grid is None else grid
		self.grid[3,3] = self.grid[4,4] = 1
		self.grid[3,4] = self.grid[4, 3] = -1

	def __getitem__(self, item):
		return self.grid[item]

	def update(self, pos, team_val, in_place=True):
		to_update = [8*x+y for x,y in self.check_lines(pos, team_val)]
		if in_place:
			self.grid.put(to_update, team_val)
		else:
			grid = copy.deepcopy(self.grid)
			grid.put(to_update, team_val)
			return Board(grid)

	def check_line(self, pos, dir, team_val):
		if self.grid[pos] != 0:
			return []
		i, j = pos
		e1, e2 = dir
		mem = []
		i += e1
		j += e2
		while 0 <= i < 8 and 0 <= j < 8 and self.grid[i, j] == -team_val:
			mem.append((i, j))
			i += e1
			j += e2
		if 0 <= i < 8 and 0 <= j < 8 and self.grid[i, j] == team_val:
			return mem
		return []

	def check_lines(self, pos, team_val):
		mem = []
		for dir in self.directions:
			mem.extend(self.check_line(pos, dir, team_val))
		return mem + [pos] if mem else []

	def is_move_possible(self, pos, team_val):
		for dir in self.directions:
			if self.check_line(pos, dir, team_val):
				return True
		return False

	def possible_moves(self, team_val):
		moves = []
		for i in range(8):
			for j in range(8):
				if self.is_move_possible((i,j), team_val):
					moves.append((i,j))
		return moves

	def execute_turn(self, pos, team_val, in_place=True):
		if not self.is_move_possible(pos, team_val):
			raise IndexError("Move {} is not allowed".format(pos))
		else:
			return self.update(pos, team_val, in_place=in_place)

	def count_score(self):
		white = - np.sum(self.grid[self.grid == -1])
		black = np.sum(self.grid[self.grid == 1])
		return black, white

	def is_in(self, pos):
		x, y = pos
		return 0 <= x <= 7 and 0 <= y <= 7

	def print(self):
		b = '\u25CE'
		w = '\u25C9'
		print('\u22BF a| b|c|d| e|f|g |h')
		for index, line in enumerate(self.grid):
			li = str(index + 1) + '|'
			for i in line:
				if i == -1:
					li += w + '|'
				elif i == 1:
					li += b + '|'
				else:
					li += '\u25A2' + '|'
			print(li)
		print(' ' + '\u203E' * 16)