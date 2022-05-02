#%%
import numpy as np
from network.nnet import nnet
from connect4.board import board
from connect4.game import Connect4

class MCTS():
    
    def __init__(self,game,nnet, num_sims = 500):
        self.num_sims = num_sims
        self.game = game
        self.nnet = nnet
        self.dp_w= {}
        self.Q = {}
        self.N = {}
        self.P = {}

    def get_probs(self):
        #Store original board config
        base_pieces = self.game.board.pieces.copy()
        for i in range(self.num_sims):
            self.search()
            self.game.board.pieces = base_pieces
        board_string = self.game.__str__()


    def search(self):
        board_string = self.game.__str__()

        #Check if winner, return -1 if so
        if board_string in self.dp_w:
            w,p,valid_actions = self.dp_w[board_string]
        else:
            valid_actions = self.game.get_valid_actions()
            w, p = self.game.get_winstate()
        if w:
            return -p
        if not any(valid_actions):
            return -1

        #Expand if leaf node
        if board_string not in self.P:
            p, v = self.nnet.expand(self.game.board.pieces)
            self.P[board_string]= p * valid_actions
            return -v

        #Evaluate edge
        max_u, best_a = -float("inf"), -1
        for a in valid_actions:
            pass
#%%
c4 = Connect4(5,5,4)
c4.get_next_state(1,3)
mcts = MCTS(c4,nnet((5,5),6,128,3))
mcts.search()
# %%