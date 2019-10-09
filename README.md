# Telegram bot that answers programming related questions searching in StackOverflow

Dialogue chatbot that:
* answers programming-related questions (using StackOverflow dataset);
* can chit-chat and simulate dialogue on all non programming-related questions.

The chit-chat mode will use a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).

Files:
* main_bot.py: bot program
* dialogue_manager.py: handles the intent recognition and the rankings
* utils.py: set of utilities to perform preorocessing and load data
* files_preparation.ipynb: notebook explaining the procedure to get the embeddings and rankings
