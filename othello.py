'''Created by Roman Malinowski '''
import copy
import numpy as np


def update(board_param, list_coord, val):
    for team_val in list_coord:
        board_param[team_val[0]][team_val[1]] = val


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


def possible_moves(board_param, team_val, avoid_moves=[], go_moves=[]):
    list_moves = []
    for i in range(8):
        for j in range(8):
            if is_move_possible(i, j, board_param, team_val):
                list_moves += [(i, j)]
    for i in avoid_moves:
        try:
            list_moves.remove(list_moves.index(i))
        except ValueError:
            pass
    temp = []
    for i in go_moves:
        try:
            temp += [list_moves.index(i)]
        except ValueError:
            pass
    if temp:
        list_moves = temp
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


def play_othello1():
    # white is -1
    # black is 1
    
    board = [[0 for i in range(8)] for j in range(8)]
    board[3][3] = board[4][4] = 1
    board[4][3] = board[3][4] = -1
    
    team = 'white'
    team_val = -1
        
    while True:
        format_ok = False
        while not format_ok:
            print('')
            print_board1(board)
            print('\nTime for the ' + team + ' team to play !')
            txt = input('Where do you want to place a pawn ?\n\n')
            print('')
            try:
                j = ord(txt[0].lower())-ord('a')
                i = int(txt[1]) - 1
            except:
                print('Use a format like "b7" !')
                continue
            if i in range(8) and j in range(8):
                if is_move_possible(i, j , board, team_val):

                    format_ok = True
                else:
                    print("You can't put it there")
            else:
                print('Use a format like "b7" !')
                
        next_move, board = execute_turn(i, j, board, team_val)
        if next_move:
            if team == 'white':
                team = 'black'
            else:
                team = 'white'
            team_val = - team_val
            continue
        elif possible_moves(board, team_val) == []:
            break
        else:
            continue
        
    black, white = count_score(board)
    print_board1(board)
    print('Final score : \nWhite Team : ' + str(white) + '\nBlack Team : ' + str(black))
    return True


def explore_board(board_move, team_val):
    moves = possible_moves(board_move[0], team_val)
    boards = []
    for move in moves:
        boards += [[check_lines(move[0], move[1], board_move[0], team_val)] + board_move[1:] + [move]]
    return boards


def plan_moves(board_param, team_val, depth=3):
    board = copy.deepcopy(board_param)
    boards = [[board]]
    for i in range(depth):
        new_boards = []
        for b in boards:
            new_boards += explore_board(b, team_val)
        team_val = - team_val
        boards = copy.deepcopy(new_boards)
    return boards


def determine_best_move(board_param, team_val, depth=3):
    board = copy.deepcopy(board_param)
    boards_moves = plan_moves(board, team_val, depth=depth)
    values = []
    moves = []
    for i in range(len(boards_moves)):
        values += [count_score(boards_moves[i][0])]
        moves += [boards_moves[i][1:]]
    if team_val == 1:
        values = [values[i][0] for i in range(len(values))]
    else:
        values = [values[i][1] for i in range(len(values))]
    return boards_moves[values.index(max(values))][1]


def alpha_beta(board_param, team_val, alpha, beta, maximize=True, depth=3):
    # We have to return the score of the root team (in the algorithm
    # the other team want to minimize this one, not maximizing it, even if
    # is equivalent in othello
    if depth == 0 or not possible_moves(board_param, team_val):
        return count_score(board_param)
    else:
        board = copy.deepcopy(board_param)
        if maximize:
            max_ = - np.inf
            for move in possible_moves(board, - team_val):
                val = alpha_beta(check_lines(move[0], move[1], board_param, -team_val), team_val, alpha, beta, not maximize, depth - 1)
                max_ = max(max_, val)
                alpha = max(alpha, max_)
                if beta <= alpha:
                    break
            return max_
        else:
            min_ = np.inf
            for move in possible_moves(board, - team_val):
                val = alpha_beta(check_lines(move[0], move[1], board_param, -team_val), team_val, alpha, beta, not maximize, depth - 1)
                min_ = min(min_, val)
                beta = min(min_, beta)
                if beta <= alpha:
                    break
            return min_


def print_board1(board):
    w = '\u25CE'
    b = '\u25C9'
    print('\u22BF a|b|c|d|e|f|g|h')

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


# w = -1
# b = 1
boardTest = [[0 for i in range(8)] for j in range(8)]
boardTest[4][4] = boardTest[3][3] = 1
boardTest[4][3] = boardTest[3][4] = -1
boardTest = check_lines(2, 4, boardTest, 1)
# sprint_board1(boardTest)
#
# boards = plan_moves(boardTest, -1, depth=5)
#
#
# print(determine_best_move(boardTest, -1, depth=5))

play_othello1()