import copy
import numpy as np


global corners
corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
global c_square
c_square = [(0, 1), (0, 7), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
global x_square
x_square = [(1, 1), (6, 6), (1, 6), (6, 1)]


def print_board1(board):
    w = '\u25CE'
    b = '\u25C9'
    print('\u22BF a| b|c|d| e|f|g |h')

    for index, line in enumerate(board):
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


def update(board_param, list_coord, val):
    for team_val in list_coord:
        board_param[team_val[0]][team_val[1]] = val


# place a piece of the team_val colour in the coordinate (i, j), and flip all the coins for that move
# return the new board
def check_lines(i, j, board_param, team_val):
    board = copy.deepcopy(board_param)
    board[i][j] = team_val

    # up vertical
    mem = []
    for v in range(i):
        if board[i - v - 1][j] == (-team_val):
            mem += [(i - v - 1, j)]
        elif board[i - v - 1][j] == 0:
            break
        else:
            update(board, mem, team_val)
            break

    # down vertical
    mem = []
    for v in range(1, 8 - i):
        if board[i + v][j] == (-team_val):
            mem += [(i + v, j)]
        elif board[i + v][j] == 0:
            break
        else:
            update(board, mem, team_val)
            break

    # left horizontal
    mem = []
    for v in range(j):
        if board[i][j - v - 1] == (-team_val):
            mem += [(i, j - v - 1)]
        elif board[i][j - v - 1] == 0:

            break
        else:
            update(board, mem, team_val)
            break

    # right horizontal
    mem = []
    for v in range(1, 8 - j):
        if board[i][j + v] == (-team_val):
            mem += [(i, j + v)]
        elif board[i][j + v] == 0:
            break
        else:
            update(board, mem, team_val)
            break

    # UL diagonal
    mem = []
    for v in range(1, min(i, j) + 1):
        if board[i - v][j - v] == (-team_val):
            mem += [(i - v, j - v)]
        elif board[i - v][j - v] == 0:
            break
        else:
            update(board, mem, team_val)
            break

    # UR diagonal
    mem = []
    for v in range(1, min(i, 8 - j - 1) + 1):
        if board[i - v][j + v] == (-team_val):
            mem += [(i - v, j + v)]
        elif board[i - v][j + v] == 0:
            break
        else:
            update(board, mem, team_val)
            break

    # DL diagonal
    mem = []
    for v in range(1, min(8 - i - 1, j) + 1):
        if board[i + v][j - v] == (-team_val):
            mem += [(i + v, j - v)]
        elif board[i + v][j - v] == 0:
            break
        else:
            update(board, mem, team_val)
            break

    # DR diagonal
    mem = []
    for v in range(1, min(8 - i, 8 - j)):
        if board[i + v][j + v] == (-team_val):
            mem += [(i + v, j + v)]
        elif board[i + v][j + v] == 0:
            break
        else:
            update(board, mem, team_val)
            break
    return board


def is_move_possible(i, j, board_param, team_val):
    board = copy.deepcopy(board_param)
    if board[i][j] == 0:
        board[i][j] = team_val
        return board != check_lines(i, j, board, team_val)
    else:
        return False


def possible_moves(board_param, team_val):
    list_moves = []
    for i in range(8):
        for j in range(8):
            if is_move_possible(i, j, board_param, team_val):
                list_moves += [(i, j)]
    return list_moves


# move a pawn and check if the opponent can move afterwards
# returns next_move (bool), board
def execute_turn(i, j, board_param, team_val):
    board = copy.deepcopy(board_param)
    board = check_lines(i, j, board, team_val)
    if not possible_moves(board, -team_val):
        return False, board
    else:
        return True, board


# return a list of elements of type [board, list of moves to get to that board]
# takes the same type of object in argument
def explore_board(board_move, team_val):
    moves = possible_moves(board_move[0], team_val)
    boards = []
    for move in moves:
        boards += [[check_lines(move[0], move[1], board_move[0], team_val)] + board_move[1:] + [move]]
    return boards


# count the score on the current board
def count_score(board):
    white = 0
    black = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                black += 1
            elif board[i][j] == -1:
                white += 1
    return black, white


# returns the number of definitive coins for each side
def definitive_coins(board_param):
    board = copy.deepcopy(board_param)
    black = 0
    for move in corners:
        x, y = move[0], move[1]
        if board[x][y] == 1:
            for i in range(8):
                for j in range(8):
                    if board[i][j] != 1:
                        break

        elif board[x][y] == -1:
            pass

    return


# we calculate the score without looking who is the player and who is the computer
# if the score is positive, is is good for the black team
# if it is negative it is good for the white team
def evaluate_score(board, team_val, turn_count, corner_value=10, x_c_value=10, possibility_value=1):
    score = 0
    # counting pieces
    for i in range(8):
        for j in range(8):
            score += board[i][j]

    # adding weights on certain moves
    for move in corners:
        score += corner_value * board[move[0]][move[1]]
    for move in x_square + c_square:
        score += - x_c_value * board[move[0]][move[1]]

    # adding weight to possibilities of move
    score += possibility_value * max(10 - turn_count, 0) * team_val * len(possible_moves(board, team_val))
    return score * team_val


def alpha_beta(board_param, team_val, alpha, beta, depth, turn_count, corner_value=10, x_c_value=10, possibility_value=1, maximize=True):
    # We have to return the score of the root team (in the algorithm
    # the other team want to minimize this one, not maximizing it, even if
    # is equivalent in othello
    if depth == 0 or not possible_moves(board_param, team_val):
        return evaluate_score(board_param, team_val, turn_count, corner_value, x_c_value, possibility_value)
    else:
        board = copy.deepcopy(board_param)
        if maximize:
            max_ = - np.inf
            team_val = - team_val
            for move in possible_moves(board, team_val):
                val = alpha_beta(check_lines(move[0], move[1], board_param, team_val), team_val, alpha, beta, depth - 1, turn_count + 1, corner_value, x_c_value, possibility_value, not maximize)
                max_ = max(max_, val)
                alpha = max(alpha, max_)
                if beta <= alpha:
                    break
            return max_
        else:
            min_ = np.inf
            team_val = - team_val
            for move in possible_moves(board, team_val):
                val = alpha_beta(check_lines(move[0], move[1], board_param, team_val), team_val, alpha, beta, depth - 1, turn_count + 1, corner_value, x_c_value, possibility_value, not maximize)
                min_ = min(min_, val)
                beta = min(min_, beta)
                if beta <= alpha:
                    break
            return min_


def determine_human(board_param, team_val):

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
            if is_move_possible(i, j, board_param, team_val):
                format_ok = True
            else:
                print("You can't put it there")
        else:
            print('Use a format like "b7" !')
    return i, j


def determine_glutton(board_param, team_val):
    moves = possible_moves(board_param, team_val)
    score = []
    for move in moves:
        black, white = count_score(check_lines(move[0], move[1], board_param, team_val))
        if team_val == 1:
            score += [black]
        else:
            score += [white]
    try:
        move = moves[score.index(max(score))]
    except ValueError:
        return 0, 0
    return move[0], move[1]


def determine_alpha_beta(board_param, team_val, turn_count, corner_value=10, x_c_value=10, possibility_value=1, depth=3):
    board = copy.deepcopy(board_param)
    list_moves = possible_moves(board, team_val)
    for move in corners:
        if move in list_moves:
            return move
    score = []
    print(list_moves)
    for move in list_moves:
        score += [alpha_beta(check_lines(move[0], move[1], board, team_val), team_val, - np.inf, np.inf, depth, turn_count + 1, corner_value, x_c_value, possibility_value, maximize=True)]
    score = [s * team_val for s in score]
    print(list_moves)
    print(score)
    return list_moves[score.index(max(score))]

