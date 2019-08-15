import numpy as np
import copy

global corners
corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
global c_square
c_square = [(0, 1), (0, 7), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
global x_square
x_square = [(1, 1), (6, 6), (1, 6), (6, 1)]


class Board:
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def __init__(self, grid=None, display_mode='basic'):
        self.grid = np.zeros((8, 8)) if grid is None else grid
        self.grid[3, 3] = self.grid[4, 4] = 1
        self.grid[3, 4] = self.grid[4, 3] = -1

        self.display_mode = display_mode

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
        if not isinstance(pos, tuple):
            return False
        for dir in self.directions:
            if self.check_line(pos, dir, team_val):
                return True
        return False

    def possible_moves(self, team_val):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.is_move_possible((i, j), team_val):
                    moves.append((i, j))
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

    def __str__(self):
        if self.display_mode == 'advanced':
            numbers = ["one1", "two1", "three1", "four1", "five1", "six1", "seven1", "eight1"]
            w = ':white_pawn:'
            b = ':black_pawn:'
            li = ':black_square_button::aletter::bletter::cletter::dletter::eletter::fletter::gletter::hletter::black_square_button:\n'

            for index, line in enumerate(self.grid):
                li += ':' + numbers[index] + ':'
                for i in line:
                    if i == -1:
                        li += w
                    elif i == 1:
                        li += b
                    else:
                        li += ':white_grid:'
                li += ':' + numbers[index] + ':' + '\n'
            li += ':black_square_button::aletter::bletter::cletter::dletter::eletter::fletter::gletter::hletter::black_square_button:\n'
            return li
        elif self.display_mode == 'basic':
            b = '\u25CE'
            w = '\u25C9'
            message = '\u22BF a| b|c|d| e|f|g |h\n'
            for index, line in enumerate(self.grid):
                li = str(index + 1) + '|'
                for i in line:
                    if i == -1:
                        li += w + '|'
                    elif i == 1:
                        li += b + '|'
                    else:
                        li += '\u25A2' + '|'
                message += li + '\n'
            return message + ' ' + '\u203E' * 16

    # returns the number of definitive coins for each side
    # rhe flag bool detects if we got into the for loop or not
    def definitive_coins(self):
        definitive = Board()
        definitive.grid[3,3] = definitive.grid[4,4] = 0
        definitive.grid[3,4] = definitive.grid[4,3] = 0

        # UL
        lim1, lim2 = 8, 8
        x, y = corners[0][0], corners[0][1]
        val = self[x, y]

        if val != 0:
            while self[x, y] == val:
                definitive.grid[x, y] = val
                flag = False
                for i in range(1, lim1):
                    flag = True
                    if self[x + i, y] == val:
                        definitive.grid[x + i, y] = val
                        continue
                    else:
                        lim1 = i - 1
                        break
                if not flag:
                    lim1 = 0

                flag = False
                for j in range(1, lim2):
                    flag = True
                    if self[x, y + j] == val:
                        definitive.grid[x, y + j] = val
                        continue
                    else:
                        lim2 = j - 1
                        break

                if not flag:
                    lim2 = 0

                lim1 -= 1
                lim2 -= 1
                if min(lim1, lim2) <= 0:
                    break
                x += 1
                y += 1

        # UR
        lim1, lim2 = 8, 8
        x, y = corners[1][0], corners[1][1]
        val = self[x, y]

        if val != 0:
            while self[x, y] == val:
                definitive.grid[x, y] = val
                flag = False
                for i in range(1, lim1):
                    flag = True
                    if self[x + i, y] == val:
                        definitive.grid[x + i, y] = val
                        continue
                    else:
                        lim1 = i - 1
                        break
                if not flag:
                    lim1 = 0

                flag = False
                for j in range(1, lim2):
                    flag = True
                    if self[x, y - j] == val:
                        definitive.grid[x, y - j] = val
                        continue
                    else:
                        lim2 = j - 1
                        break
                if not flag:
                    lim2 = 0

                lim1 -= 1
                lim2 -= 1
                if min(lim1, lim2) <= 0:
                    break
                x += 1
                y -= 1

        # DL
        lim1, lim2 = 8, 8
        x, y = corners[2][0], corners[2][1]
        val = self[x, y]
        if val != 0:
            while self[x, y] == val:
                definitive.grid[x, y] = val
                flag = False
                for i in range(1, lim1):
                    flag = True
                    if self[x - i, y] == val:
                        definitive.grid[x - i, y] = val
                        continue
                    else:
                        lim1 = i - 1
                        break
                if not flag:
                    lim1 = 0
                flag = False

                for j in range(1, lim2):
                    flag = True
                    if self[x, y + j] == val:
                        definitive.grid[x, y + j] = val
                        continue
                    else:
                        lim2 = j - 1
                        break
                if not flag:
                    lim2 = 0

                lim1 -= 1
                lim2 -= 1
                if min(lim1, lim2) <= 0:
                    break
                x -= 1
                y += 1

        # UR
        lim1, lim2 = 8, 8
        x, y = corners[3][0], corners[3][1]
        val = self[x, y]
        if val != 0:
            while self[x, y] == val:
                flag = False
                definitive.grid[x, y] = val
                for i in range(1, lim1):
                    flag = True
                    if self[x - i, y] == val:
                        definitive.grid[x - i, y] = val
                        continue
                    else:
                        lim1 = i - 1
                        break
                if not flag:
                    lim1 = 0
                flag = False

                for j in range(1, lim2):
                    flag = True
                    if self[x, y - j] == val:
                        definitive.grid[x, y - j] = val
                        continue
                    else:
                        lim2 = j - 1
                        break
                if not flag:
                    lim2 = 0

                lim1 -= 1
                lim2 -= 1
                if min(lim1, lim2) <= 0:
                    break
                x -= 1
                y -= 1

        return definitive

    def compute_score(self, team_val, maximize):
        black, white = self.count_score()
        disc_diff = 100 * (black - white) / (black + white)

        black_move, white_move = len(self.possible_moves(1)), len(self.possible_moves(-1))
        move_diff = 100 * (black_move - white_move) / (black_move + white_move + 1)

        black_corner = white_corner = 0
        for corner in corners:
            if self[corner[0], corner[1]] == 1:
                black_corner += 1
            elif self[corner[0], corner[1]] == -1:
                white_corner += 1
        corner_diff = 100 * (black_corner - white_corner) / (black_corner + white_corner + 1)

        definitive = self.definitive_coins()
        black_def, white_def = definitive.count_score()
        def_diff = 100 * (black_def - white_def) / (black_def + white_def + 1)

        c_black = c_white = 0
        for i, move in enumerate(c_square):
            if self[move] == 1 and corners[i // 2] != 1 and definitive[move] != 1:
                c_black += 1
            if self[move] == -1 and corners[i // 2] != -1 and definitive[move] != -1:
                c_white += 1
        c_diff = 100 * (c_black - c_white) / (c_black + c_white + 1)

        x_black = x_white = 0
        for i, move in enumerate(x_square):
            if self[move] == 1 and corners[i] != 1 and definitive[move] != 1:
                x_black += 1
            if self[move] == -1 and corners[i] != -1 and definitive[move] != -1:
                x_white += 1
        x_diff = 100 * (x_black - x_white) / (x_black + c_white + 1)

        turn = black + white

        score = 2 * move_diff * max(1, 12 - turn) + 100 * corner_diff - 100 * x_diff - 50 * c_diff + 30 * def_diff + disc_diff * max(0, turn - 58) * 100

        if turn == 64:
            score = disc_diff * 100

        if maximize:
            score = - score * team_val
        else:
            score = score * team_val

        return score


    def __eq__(self, board):
        if isinstance(board, Board):
            return (self.grid == board.grid).all()
        else:
            return super().__eq__(board)


class Game:
    def __init__(self, player1, player2, display_func=print, board=None, cur_team=-1):
        self.player1 = player1
        self.player1.set_team('white')
        self.player2 = player2
        self.player2.set_team('black')
        self.board = Board() if board == None else board
        self.cur_team = cur_team
        self.display_func = display_func
        if self.display_func:
            self.display_func('Starting a new game ! {} vs {}'.format(player1, player2))
            self.display_func(self.board)

    def next(self):
        if not self.board.possible_moves(self.cur_team):
            if not self.board.possible_moves(-self.cur_team):
                return self.end()
            else:
                if self.display_func:
                    self.display_func('{} cannot play'.format('White' if self.cur_team == -1 else 'Black'))
                self.cur_team = -self.cur_team
        else:
            if self.cur_team == -1:
                move = self.player1.play(self.board)
            else:
                move = self.player2.play(self.board)
            self.board.execute_turn(move, self.cur_team)
            if self.display_func:
                self.display_func('{} played {}'.format('White' if self.cur_team == -1 else 'Black',
                                                        chr(move[1] + ord('a')) + str(move[0] + 1)))
                self.display_func(self.board)
            self.cur_team = -self.cur_team

    def end(self):
        black, white = self.board.count_score()
        if self.display_func:
            self.display_func('Game is finished. Final score : \nWhite Team ({}) : {}\nBlack team ({}) : {}'.format(self.player1, white, self.player2, black))
        return black, white

    def rollout(self):
        scores = None
        while not scores:
            scores = self.next()
        return scores

