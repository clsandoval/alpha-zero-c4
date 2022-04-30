import numpy as np 
import sys 
sys.path.append('..')


class board():
    
    def __init__(self,height, width, win_length,pieces):
        self.height = height
        self.width = width
        self.win_length = win_length
        if self.pieces is not None:
            self.pieces = pieces
        else:
            self.pieces = np.zeros((self.height, self.width),dtype=np.int)
    
    def take_action():
        pass
    
    def __str__(self):
        return str(self.pieces)