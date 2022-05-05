import numpy as np
from mcts import MCTS

class Coach():
    
    def __init__(self,nnet, game,num_sims, num_eps):
        self.game = game
        self.nnet=nnet 
        self.mcts = MCTS(game,nnet,num_sims)
        self.num_eps = num_eps

    def episode():
        pass

    def train(self):
        pass