#%%
import numpy as np
import math
from network.nnet import nnet
from connect4.board import board
from connect4.game import Connect4

class MCTS():
    
    def __init__(self,game,nnet, num_sims = 500,cpuct =  3.0):
        self.num_sims = num_sims
        self.game = game
        self.nnet = nnet
        self.dp_w= {}
        self.Q = {}
        self.Nsa = {}
        self.N = {}
        self.P = {}
        self.cpuct = cpuct

    def get_probs(self, temp = 10):
        """
        Returns action probabilities of length (action_size)
        """
        #Store original board config
        state = self.game.__str__()
        base_pieces = self.game.board.pieces.copy()
        for i in range(self.num_sims):
            self.search()
            self.game.board.pieces = base_pieces.copy()

        #return exponentiated counts of state actions visited
        counts = [self.Nsa[(state,a)] if (state,a) in self.Nsa else 0 for a in range(self.game.get_action_size())]
        counts = [x ** (1./temp) for x in counts]
        count_sum = float(sum(counts))
        probs = [x/count_sum for x in counts]
        return probs
        

    def search(self):
        """ 
        Execute one monte carlo tree search returning the value at the given state
        """
        state = self.game.__str__()

        #Check if winner, return -1 if so
        if state in self.dp_w:
            win,player,valid_actions = self.dp_w[state]
        else:
            valid_actions = self.game.get_valid_actions()
            win, player = self.game.get_winstate()
        if win:
            return -player
        if not any(valid_actions):
            return -1

        #Expand if leaf node
   
        if state not in self.P:
            pi, value = self.nnet.expand(self.game.board.pieces)
            self.P[state]= pi * valid_actions
            self.N[state] = 0
            return -value

        #Evaluate edge
        max_q_u, best_a = -float("inf"), -1
        valid_moves = [i for i, x in enumerate(valid_actions) if x]
        for action in valid_moves:
            state_action = (state,action)
            if state_action not in self.Q:
                q_u = self.cpuct * self.P[state][action] * math.sqrt(self.N[state] + .00000001) 
            else: 
                u = self.cpuct * self.P[state][action] * math.sqrt(self.N[state])/(1+self.Nsa[(state,action)]) 
                q_u = self.Q[state_action] + u
            if q_u > max_q_u:
                max_q_u = q_u
                best_a = action

        #take action and pass to next player
        self.game.board.take_action(1,best_a)
        value = self.search()
        #backup
        if (state,best_a) not in self.Q:
            self.Q[(state,best_a)] = value
            self.Nsa[(state,best_a)] = 1 
        else:
            self.Nsa[(state,best_a)] += 1
            self.Q[(state,best_a)] = (value + self.Nsa[(state,best_a)] * self.Q[(state,best_a)]) / (self.Nsa[(state,best_a)] + 1) 
        self.N[state] += 1
        return -value
    
                    
# %%

# %%
