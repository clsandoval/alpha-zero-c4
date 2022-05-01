#%%
from random import Random
from connect4.game import Connect4
from connect4.players import RandomPlayer

#%%
p1 = RandomPlayer()
p2 = RandomPlayer()
# %%
for i in range(1000):
    c4 = Connect4(5,5,4)
    player_counter = 0
    while True:
        if player_counter %2: player = p1 
        else: player = p2
        valid_actions = c4.get_valid_actions()
        if all(False == valid_actions):
            c4.display_self()
            break
        action = player.decision(c4.board,valid_actions)

        next_state = c4.get_next_state(1,action)
        win, winning_player = c4.get_winstate()
        if win: 
            print(player_counter%2,winning_player)
            c4.display_self()
            break
        c4.board.pieces = next_state * -1
        player_counter += 1

# %%
