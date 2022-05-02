import numpy as np 
import sys 


class RandomPlayer():
    """ Random Agent for testing purposes and benchmarks"""

    def __init__(self):
        pass

    def decision(self, board, valid_actions):
        valid_moves = [i for i, x in enumerate(valid_actions)) if x]
        return np.random.choice(valid_actions)        