import copy
import numpy as np
from Board import Board

global corners
corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
global c_square
c_square = [(0, 1), (1, 0), (1, 7), (0, 6), (7, 1), (6, 0), (6, 7), (7, 6)]
global x_square
x_square = [(1, 1), (1, 6), (6, 1), (6, 6)]


def compute_score(board, team_val, maximize):
    # black, white = board.count_score()
    # disc_diff = 100 * (black - white) / (black + white)

    black_move, white_move = len(board.possible_moves(1)), len(board.possible_moves(-1))
    move_diff = 100 * (black_move - white_move) / (black_move + white_move + 1)

    black_corner = white_corner = 0
    for corner in corners:
        if board[corner[0], corner[1]] == 1:
            black_corner += 1
        elif board[corner[0], corner[1]] == -1:
            white_corner += 1
    corner_diff = 100 * (black_corner - white_corner) / (black_corner + white_corner + 1)

    c_black = c_white = 0
    for i, move in enumerate(c_square):
        if board[move] == 1 and corners[i//2] != 1:
            c_black += 1
        if board[move] == -1 and corners[i//2] != -1:
            c_white += 1
    c_diff = 100 * (c_black - c_white) / (c_black + c_white + 1)

    x_black = x_white = 0
    for i, move in enumerate(x_square):
        if board[move] == 1 and corners[i] != 1:
            x_black += 1
        if board[move] == -1 and corners[i] != -1:
            x_white += 1
    x_diff = 100 * (x_black - x_white) / (x_black + c_white + 1)

    # definitive = board.definitive_coins()
    # for move in x_square + c_square:
    #     if definitive[move] != board[move]:
    #         definitive.grid[move] = - board[move]
    # black_def, white_def = definitive.count_score()
    # def_diff = 100 * (black_def - white_def) / (black_def + white_def + 1)

    score = 2 * move_diff + 100 * corner_diff - 100 * x_diff - 50 * c_diff

    if maximize:
        score = score * team_val
    else:
        score = - score * team_val
    return score

    return score


def evaluate_position(board, team_val, maximize):
    val_board = [[500 , -300, 100,  50,  50, 100, -300,  500],
                [-300, -500, -50, -50, -50, -50, -500, -300],
                [100 ,  -50, 100,   0,   0, 100,  -50,  100],
                [50  ,  -50,   0,   0,   0,   0,  -50,   50],
                [50  ,  -50,   0,   0,   0,   0,  -50,   50],
                [100 ,  -50, 100,   0,   0, 100,  -50,  100],
                [-300, -500, -50, -50, -50, -50, -500, -300],
                [500 , -300, 100,  50,  50, 100, -300,  500]]

    score = 0

    for i in range(8):
        for j in range(8):
            score += val_board[i][j] * board[i, j]
    if maximize:
        score = - score * team_val
    else:
        score = score * team_val
    return score


def alpha_beta(board, team_val, depth, alpha, beta, maximize=True):
    # We have to return the score of the root team (in the algorithm
    # the other team want to minimize this one, not maximizing it, even if
    # is equivalent in othello
    if depth == 0 or not board.possible_moves(team_val):
        # return evaluate_score(board_param, team_val, turn_count, maximize)
        return compute_score(board, team_val, maximize)
    else:
        team_val = - team_val
        if maximize:
            max_ = - np.inf
            for move in board.possible_moves(team_val):
                val = alpha_beta(board.update(move, team_val, in_place=False), team_val, depth - 1, alpha, beta, not maximize)
                max_ = max(max_, val)
                alpha = max(alpha, max_)
                if alpha >= beta:
                    break
            return max_
        else:
            min_ = np.inf
            for move in board.possible_moves(team_val):
                val = alpha_beta(board.update(move, team_val, in_place=False), team_val, depth - 1, alpha, beta, not maximize)
                min_ = min(min_, val)
                beta = min(beta, min_)
                if alpha >= beta:
                    break
            return min_


def determine_human(board, team_val):
    if team_val == 1:
        team = 'black'
    else:
        team = 'white'
    format_ok = False
    while not format_ok:
        print('\nTime for the ' + team + ' team to play !')
        txt = input('Where do you want to place a pawn ?\n\n')
        print('')
        try:
            j = ord(txt[0].lower()) - ord('a')
            i = int(txt[1]) - 1
        except:
            print('Use a format like "b7" !')
            continue
        if i in range(8) and j in range(8):
            if board.is_move_possible((i, j), team_val):
                format_ok = True
            else:
                print("You can't put it there")
        else:
            print('Use a format like "b7" !')
    return i, j


def determine_alpha_beta(board, team_val, depth=3):
    list_moves = board.possible_moves(team_val)
    for move in corners:
        if move in list_moves:
            return move
    score = []
    for move in list_moves:
        score += [alpha_beta(board.update(move, team_val, in_place=False), team_val, depth, alpha=-np.inf, beta=np.inf, maximize=True)]
    print(list_moves, score)
    return list_moves[score.index(max(score))]
