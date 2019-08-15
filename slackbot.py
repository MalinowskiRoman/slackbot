import slack
import random
import requests
import datetime
import time
import praw
import calendar
import torch
from Board import Board, Game
from Agents import MLAgent, DiggingGlutton, Player, DenseBrain, AlphaBeta

with open('token.txt', 'r') as f:
    token = f.readline()
    token = token[:-1]
    bot_token = f.readline()

channel_name: str = 'general'


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
        response_channel = requests.get(url)
        history = response_channel.json()
        return history
    except KeyError:
        raise Exception('Could not find the name of the channel : {}'.format(name))
    except ValueError as err:
        print(err)


def get_user_name(user, tok=token):
    try:
        url = 'https://slack.com/api/users.info?token=' + tok + '&user=' + user
        response_name = requests.get(url)
        response_name = response_name.json()
        return response_name['user']['real_name']
    except KeyError:
        raise Exception('Could not find the name of the user : {}'.format(user))
    except ValueError as err:
        print(err)


def write_message(message):
    try:
        response_message = requests.get(
            'https://slack.com/api/chat.postMessage?token=' + bot_token + '&channel=' + channel_name + '&text=' +
            str(message) + '&pretty=1')
        print(response_message.json())
    except ValueError as err:
        print(err)

def message_sin_call(text, call):
    length = len(call)
    for i in range(len(text) - length):
        if text[i:i + length] == call:
            return text[:i] + text[i + length:]
    return "Pour etre tout a fait franc, j'ai pas compris"


def analyse_message(tok=token):
    # try:
    history = get_history_channel(tok)
    try:
        last_message = history['messages'][0]
        try:
            last_message['bot_id']
        except KeyError:
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
                with open('C:/Users/roman/Desktop/Compteur.txt', 'r') as count_file:
                    n = count_file.readline()
                    m = count_file.readline()
                    write_message(n + m)
            elif 'othello' in text.lower():
                with open('othello.txt', 'w') as player_file:
                    player_file.writelines(last_message['user'] + '\n')
                if text.lower() == 'othello':
                    write_message('Deuxième joueur ? (Ecrire \"othello joueur 2\")')
                else:
                    player_number = 2
                    if 'player' in text.lower():
                        try:
                            player_number = int(text.lower()[text.lower().find('player') + 7])
                            if player_number not in [1, 2]:
                                player_number = 2
                        except IndexError:
                            player_number = 2
                        except ValueError:
                            player_number = 2

                    if 'glutton' in text.lower():
                        try:
                            k = int(text.lower()[text.lower().find('glutton') + 8])
                        except IndexError:
                            k = 2
                        except ValueError:
                            k = 2
                        player2 = DiggingGlutton(depth=k)
                    elif 'learning' in text.lower():
                        brain = DenseBrain(128)
                        brain.load_state_dict(torch.load('models/against_self_and_alpha_beta_44.pt'))
                        player2 = MLAgent(brain)
                    elif 'alpha' in text.lower():
                        try:
                            k = int(text.lower()[text.lower().find('alpha') + 6])
                        except IndexError:
                            k = 2
                        except ValueError:
                            k = 2
                        player2 = AlphaBeta(k)
                    else:
                        player2 = DiggingGlutton(depth=2)
                    player1 = SlackPlayer(last_message['user'])
                    # if we choose to let the IA Begin
                    if player_number == 1:
                        player1, player2 = player2, player1
                    board = Board(display_mode='advanced')
                    game = Game(player1, player2, board=board, display_func=write_message)
                    game.rollout()

            elif 'othello joueur 2' in text.lower():
                with open('othello.txt', 'a') as player_file:
                    player_file.writelines(last_message['user'])
                with open('othello.txt', 'r') as player_file:
                    player1 = player_file.readline()[:-1]
                    player2 = player_file.readline()
                board = Board(display_mode='advanced')
                player1 = SlackPlayer(player1)
                player2 = SlackPlayer(player2)
                game = Game(player1, player2, board=board, display_func=write_message)
                game.rollout()

            elif 'help' in text.lower():
                write_message(
                    'Petit rappel des commandes :\nPour que je recopie bizarrement votre message, écrivez "@Bob'
                    ' mon message"\nPour rajouter un point au compteur de vouvoiement, écrivez "Kubat" ainsi que '
                    '"Nico" ou "Maxime" dans le même message. J\'ignore les majuscules :wink: \nPour commencer une'
                    ' partie d\'Othello contre un autre membre du slack, écrivez "othello", puis votre adversaire '
                    'écrit "othello joueur 2". \nPour jouer contre une IA, écrivez "othello" suivi de '
                    '"glutton \'k\' " pour un glouton de profondeur k, "learning" pour une IA de reinforcement '
                    'learning, "alpha \'k\' " pour une IA de recherche arborescent de profondeur k. \nAjoutez '
                    '"player 1" pour laisser l\'IA commencer, sinon "player 2" (par défaut vous commencez).'
                    '\n Pour les mouvements, écrivez simplement les coordonnées "b5".\n Pour relire ce message, '
                    'écrivez "help"')
    except IndexError as err:
        print(err)
    except KeyError as err:
        print(err)
        # except KeyError as err:
        #     print(history)
        #     print(err)


class SlackPlayer(Player):
    def __init__(self, user, team=None):
        super().__init__(team)
        self.user = user
        self.name = get_user_name(user)

    def play(self, board):
        write_message('\nTeam ' + self.team + ', where do you want to place a pawn ?\n')
        while True:
            try:
                time.sleep(1)
                history = get_history_channel(token)
                last_message = history['messages'][0]
                try:
                    last_message['bot_id']
                except KeyError:
                    user = last_message['user']
                    if user != self.user:
                        write_message("C'est pas à toi de jouer wesh")
                    else:
                        # la partie qui nous interesse
                        text = last_message['text']
                        if len(text) == 2:
                            try:
                                j = ord(text[0].lower()) - ord('a')
                                i = int(text[1]) - 1
                                if board.is_move_possible((i, j), self.team_val):
                                    return i, j
                                else:
                                    write_message('You can\'t put it there!')
                            except IndexError:
                                write_message('Use a format like "b7" !')
                            except ValueError:
                                write_message('Use a format like "b7" !')
                        elif 'quit' in text.lower():
                            write_message('Game ended')
                            return 'quit', 0
            except IndexError:
                print('Pas de messages while othello')
                return -1, -1
            except KeyError as err:
                print(err)
                return -1, -1


def update_counter(name):
    with open('C:/Users/roman/Desktop/Compteur.txt', 'r') as count_file:
        n = count_file.readline()
        m = count_file.readline()
        if name.lower() == 'nico':
            res = int(n[-2]) + 1
            n = n[:-2] + str(res) + '\n'
            write_message(transform("C'est noté !") + '\n' + n + m)
        elif name.lower() == 'maxime':
            res = int(m[-2]) + 1
            m = m[:-2] + str(res) + '\n'
            write_message(transform("C'est noté !") + '\n' + n + m)
        else:
            write_message(transform("J'arrive pas à lire wesh"))
    with open('C:/Users/roman/Desktop/Compteur.txt', 'w') as count_file:
        count_file.writelines(n + m)


'''Slack connection and setup'''


user_account = 'xoxp-684260139683-689311425137-684247936882-151caac50e5f63ad69b6272127ade6a3'
bot_id = '<@ULHPSPLH4>'
response = requests.get('https://slack.com/api/conversations.list?token=' + bot_token)
channels = response.json()
print(channels)
for dic in channels['channels']:
    if dic['name'] == channel_name:
        channel_id = dic['id']
        print('Channel id : ' + channel_id)

# install slackclient if WebClient is not recognized
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
reddit = praw.Reddit(client_id='oAS5--tyAeKDvg',
                     client_secret='HFYIvgFoe9LyKH_JqLPeMW3XmtE',
                     user_agent='redditBotScraper',
                     username='slackSmartBuild',
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
                    ' partie d\'Othello contre un autre membre du slack, écrivez "othello", puis votre adversaire '
                    'écrit "othello joueur 2". \nPour jouer contre une IA, écrivez "othello" suivi de '
                    '"glutton \'k\' " pour un glouton de profondeur k, "learning" pour une IA de reinforcement '
                    'learning, "alpha \'k\' " pour une IA de recherche arborescent de profondeur k. \nAjoutez '
                    '"player 1" pour laisser l\'IA commencer, sinon "player 2" (par défaut vous commencez).'
                    '\n Pour les mouvements, écrivez simplement les coordonnées "b5".\n Pour relire ce message, '
                    'écrivez "help"')
                flag = False
        if (hour, minute) == (10, 31):
            flag = True

        if (hour, minute) == (18, 0):
            if flag:
                write_message(transform("La journééééeeee est finiiiiie\nVous êtes officellement autorisés à jouer à "
                                        "Smash.\nQue le meilleur gagne !"))
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
