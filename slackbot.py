import slack
import random
import requests
import datetime
import time
import praw
import calendar
from othello import *

token = 'xoxp-684260139683-689311425137-684247936882-151caac50e5f63ad69b6272127ade6a3'
bot_token = 'xoxb-684260139683-697808802582-aBdjKsw9s19ikbcFrtPpwEP7'

channel_name = 'general'
othello_flag = False
last_move = ''
move = ''
team = 'white'


def transform(message):
    res = message[0].lower()
    try:
        for char in message.lower()[1:]:
            if char.isalpha():
                if random.randrange(2) == 0:
                    res += char.upper()
                else:
                    res += char
            else:
                res += char
        return res
    except IndexError:
        return message


def get_history_channel(name, tok=token):
    try:
        global channel_id
        url = 'https://slack.com/api/conversations.history?token=' + tok + '&channel=' + channel_id
        response = requests.get(url)
        history = response.json()
        return history
    except:
        raise Exception('Could not find the name of the channel : {}'.format(name))


def write_message(message):
    response = requests.get(
        'https://slack.com/api/chat.postMessage?token=' + bot_token + '&channel=' + channel_name + '&text=' + message + '&pretty=1')
    print(response.json())


def message_sin_call(text, call):
    length = len(call)
    for i in range(len(text) - length):
        if text[i:i + length] == call:
            return text[:i] + text[i + length:]
    return "Pour etre tout a fait franc, j'ai pas compris"


def analyse_message(tok=token):
    try:
        history = get_history_channel(tok)
        try:
            last_message = history['messages'][0]
            try:
                last_message['bot_id']
            except:
                # la partie qui nous interesse
                text = last_message['text']
                if bot_id in text:
                    message_clean = message_sin_call(text, bot_id)
                    if message_clean[0] == ' ' and len(message_clean) > 1:
                        message_clean = message_clean[1:]
                    write_message(transform(message_clean))
                elif 'kubat' in text.lower() and 'maxime' in text.lower():
                    update_counter('maxime')
                elif 'kubat' in text.lower() and 'nico' in text.lower():
                    update_counter('nico')
                elif 'affiche' in text.lower() and 'score' in text.lower():
                    with open('C:/Users/roman/Desktop/Compteur.txt', 'r') as f:
                        n = f.readline()
                        m = f.readline()
                        write_message(n + m)
                elif 'play othello' in text.lower():
                    with open('C:/Users/roman/Desktop/othello.txt', 'w') as f:
                        f.writelines(last_message['user'] + '\n')
                    write_message('Deuxième joueur ? (Ecrire othello joueur 2)')
                elif 'othello joueur 2' in text.lower():
                    with open('C:/Users/roman/Desktop/othello.txt', 'a') as f:
                        f.writelines(last_message['user'])
                    global othello_flag
                    othello_flag = True
                    play_othello()
                elif 'help' in text.lower():
                    write_message(
                        'Petit rappel des commandes :\nPour que je recopie bizarrement votre message, écrivez "@Bob'
                        ' mon message"\nPour rajouter un point au compteur de vouvoiement, écrivez "Kubat" ainsi que '
                        '"Nico" ou "Maxime" dans le même message. J\'ignore les majuscules :wink: \nPour commencer une'
                        ' partie d\'Othello écrivez "play othello", puis le deuxième joueur écrit "othello joueur 2".\n Pour'
                        ' les mouvements, écrivez simplement les coordonnées "b5".\n Pour relire ce message, écrivez'
                        ' "help"')
        except IndexError:
            print('Pas de messages')
        except KeyError:
            print('Key Error')
    except Exception:
        print('COULD NOT FIND THE NAME OF THE CHANNEL')
        # print(history)



def analyse_othello_move(team, tok=token):
    history = get_history_channel(tok)
    try:
        last_message = history['messages'][0]
        try:
            last_message['bot_id']
        except:
            user = last_message['user']
            with open('C:/Users/roman/Desktop/othello.txt', 'r') as f:
                player1 = f.readline()[:-1]
                player2 = f.readline()
            if team == 'white':
                player_username = player1
            else:
                player_username = player2
            # la partie qui nous interesse
            text = last_message['text']
            if len(text) == 2:
                if user != player_username:
                    write_message("C'est pas à toi de jouer wesh")
                try:
                    j = ord(text[0].lower()) - ord('a')
                    i = int(text[1]) - 1
                    global move
                    move = text
                    return i, j
                except:
                    write_message('Use a format like "b7" !')
            elif 'quit' in text:
                return 'quit', 0
        return -1, -1
    except IndexError:
        print('Pas de messages while othello')
        return -1, -1
    except KeyError:
        print('Key Error while othello')
        return -1, -1


def print_board(board):
    numbers = ["one1", "two1", "three1", "four1", "five1", "six1", "seven1", "eight1"]
    w = ':white_pawn:'
    b = ':black_pawn:'
    li = ':black_square_button::aletter::bletter::cletter::dletter::eletter::fletter::gletter::hletter:\n'

    for index, line in enumerate(board):
        li += ':' + numbers[index] + ':'
        for i in line:
            if i == -1:
                li += w
            elif i == 1:
                li += b
            else:
                li += ':white_grid:'
        li += '\n'
    write_message(li)


def update_counter(name):
    with open('C:/Users/roman/Desktop/Compteur.txt', 'r') as f:
        n = f.readline()
        m = f.readline()
        if name.lower() == 'nico':
            res = int(n[-2]) + 1
            n = n[:-2] + str(res) + '\n'
            write_message(transform("C'est noté !") + '\n' + n + m)
        elif name.lower() == 'maxime':
            res = int(m[-2]) + 1
            m = m[:-2] + str(res) + '\n'
            write_message(transform("C'est noté !") + '\n' + n + m)
        else:
            write_message(transform("J'arrive pas a lire wesh"))
    with open('C:/Users/roman/Desktop/Compteur.txt', 'w') as f:
        f.writelines(n + m)


def play_othello():
    # white is -1
    # black is 1

    board = [[0 for i in range(8)] for j in range(8)]
    board[3][3] = board[4][4] = 1
    board[4][3] = board[3][4] = -1
    global team
    team = 'white'
    pos = -1
    while True:
        format_ok = False
        while not format_ok:
            print_board(board)
            write_message('\nTeam ' + team + ', where do you want to place a pawn ?\n\n')
            write_message('')
            global move
            global last_move
            while move == last_move:
                i, j = analyse_othello_move(team)
                if i > -1:
                    last_move = move
                    break
                if i == 'quit':
                    black, white = count_score(board)
                    print_board(board)
                    write_message('Final score : \nWhite Team : ' + str(white) + '\nBlack Team : ' + str(black))
                    global othello_flag
                    othello_flag = False
                    return 0
            if i in range(8) and j in range(8):
                if is_move_possible(i, j, board, pos):
                    format_ok = True
                    # print(format_ok)
                    continue
                else:
                    write_message("You can't put it there")
        next_move, board = execute_turn(i, j, board, pos)
        if next_move:
            if team == 'white':
                team = 'black'
            else:
                team = 'white'
            pos = - pos
            continue
        elif possible_moves(board, pos) == []:
            print('No possible moves !')
            break
        else:
            continue
    black, white = count_score(board)
    print_board(board)
    write_message('Final score : \nWhite Team : ' + str(white) + '\nBlack Team : ' + str(black))
    global othello_flag
    othello_flag = False
    return 0


'''Slack connection and setup'''

token = 'xoxp-684260139683-689311425137-684247936882-151caac50e5f63ad69b6272127ade6a3'
bot_token = 'xoxb-684260139683-697808802582-aBdjKsw9s19ikbcFrtPpwEP7'
user_account = 'xoxp-684260139683-689311425137-684247936882-151caac50e5f63ad69b6272127ade6a3'
bot_id = '<@ULHPSPLH4>'

response = requests.get('https://slack.com/api/conversations.list?token=' + bot_token)
channels = response.json()
for dic in channels['channels']:
    if dic['name'] == channel_name:
        global channel_id
        channel_id = dic['id']
        print('Channel id : ' + channel_id)

client = slack.WebClient(token=bot_token)
if client.rtm_connect():
    print("Starter Bot connected and running!")
    # Read bot's user ID by calling Web API method `auth.test`
    starterbot_id = client.api_call("auth.test")["user_id"]
    bot_id = '<@' + starterbot_id + '>'
    print('Bot ID : ' + bot_id)
else:
    print("Connection failed. Exception traceback printed above.")

'''Reddit connection and setup'''
reddit = praw.Reddit(client_id='oAS5--tyAeKDvg', \
                     client_secret='HFYIvgFoe9LyKH_JqLPeMW3XmtE', \
                     user_agent='redditBotScraper', \
                     username='slackSmartBuild', \
                     password='slackSmartBuild')
subreddit = reddit.subreddit('copypasta')
new_subreddit = subreddit.new()

flag = True
now = datetime.datetime.now()
mem = now.hour
while True:
    # try:
    analyse_message()

    now = datetime.datetime.now()
    day = calendar.weekday(now.year, now.month, now.day)

    hour, minute = now.hour, now.minute

    if day not in ['Saturday', 'Sunday']:
        if hour in range(10, 18) and mem != hour:
            mem = hour
            for submission in subreddit.new():
                write_message(submission.title + '\n' + '\n' + submission.selftext)
                break

        if (hour, minute) == (16, 36):
            if flag:
                write_message('@Mathieu @Vadim' + transform("  Ecrivez un mail pour l'hebreu bande de batards !"))
                flag = False
        if (hour, minute) == (16, 37):
            flag = True

        if (hour, minute) == (12, 30):
            if flag:
                write_message(transform("On va manger ? J'ai faaaaaim :pickle:"))
                flag = False
        if (hour, minute) == (12, 31):
            flag = True

        if (hour, minute) == (10, 30):
            if flag:
                write_message(
                    'Petit rappel des commandes :\nPour que je recopie bizarrement votre message, écrivez "@Bob'
                    ' mon message"\nPour rajouter un point au compteur de vouvoiement, écrivez "Kubat" ainsi que '
                    '"Nico" ou "Maxime" dans le même message. J\'ignore les majuscules :wink: \nPour commencer une'
                    ' partie d\'Othello écrivez "play othello", puis le deuxième joueur écrit "othello moi".\n Pour'
                    ' les mouvements, écrivez simplement les coordonnées "b5".\n Pour relire ce message, écrivez'
                    ' "help"')
                flag = False
        if (hour, minute) == (10, 31):
            flag = True

        if (hour, minute) == (18, 0):
            if flag:
                write_message(transform("La journééééeeee est finiiiiie"))
                flag = False
        if (hour, minute) == (18, 1):
            flag = True

        if (hour, minute) == (9, 0):
            if flag:
                write_message(transform("La journée commence, on prend son café et on y va !"))
                flag = False
        if (hour, minute) == (9, 1):
            flag = True
        time.sleep(0.5)
    # except:
    #     print('Error somewhere')
    #     time.sleep(0.5)
