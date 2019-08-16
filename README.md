# README

This repo contains code for a slackbot, which can scrap reddit posts, display reminders, and most importantly:
### Offers the possibility to play Othello in Slack!
Othello is a strategy game *similar* to Go. We created different AI (alpha-beta tree search, trained with reinforcement learning etc) to play against. 

- `models/` contains the trained models for the AI
- `Agents.py` contains different classes defining the different AI available.
- `Board.py` contains the class used to play Othello, *i.e.* the definition of the board, possible moves, score computation etc.
- `othello.py` contains utility functions for creating tournaments between different AI
- `slackbot.py` contains the code for the slackbot. It is heavily personalized and designed for students with too much time on their hands, so please be indulgent. Also, it cannot be used as such, you need to generate tokens from slack and Reddit for it to work! 
