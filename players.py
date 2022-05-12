import numpy as np 
import sys 
from mcts import MCTS  



class NetworkPlayer():
    """Player utilizing an AlphaZero agent"""
    def __init__(self,nnet,name="Network Agent"):
        self.nnet = nnet
        self.name = name
    
    def init_game(self,game):
        self.game = game
        self.mcts = MCTS(self.game,self.nnet)

    def decision(self,board,valid_actions):
        probs = self.mcts.get_probs()
        actions = np.arange(len(probs))
        return np.random.choice(actions,p=probs)    

class RandomPlayer():
    """ Random Agent for testing purposes and benchmarks"""

    def __init__(self,name="Random Agent"):
        self.name = name
        pass

    def init_game(self,game):
        self.game = game

    def decision(self, board, valid_actions):
        valid_moves = [i for i, x in enumerate(valid_actions) if x]
        return np.random.choice(valid_moves)        

class HumanPlayer():
    """Human controlled player"""

    def __init__(self):
        pass 

    def init_game(self,game):
        self.game = game 
    
    def decision(self,board,valid_actions):
        print(board.__str__())
        a = input()
        return int(a)