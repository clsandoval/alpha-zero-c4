import numpy as np 
from connect4.logic import get_row
import sys 
sys.path.append('..')


"""
board class implementing the connect 4 game board
"""
class board():
    
    def __init__(self,height, width, win_length,pieces):
        self.height = height
        self.width = width
        self.win_length = win_length
        if pieces is not None:
            self.pieces = pieces
        else:
            self.pieces = np.zeros((self.height, self.width),dtype=np.int)
    
    def take_action(self,player,column,pieces):
        row = get_row(pieces,column)
        if row == -1:
            raise ValueError("no valid move at {} {}".format(row,column))
        pieces[row][column] = player
        pieces *=-1
        return pieces
        
    
    def __str__(self):
        return str(self.pieces)