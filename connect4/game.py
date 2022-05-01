import numpy as np 
import sys 
from connect4.board import board
from connect4.logic import get_valid_actions as gva
from connect4.logic import get_winstate as gws
sys.path.append('..')


class Connect4():
    """ 
    Connect4 class implementing the connect4 game engine
    """

    def __init__(self,height, width, win_length,pieces=None):
        self.board = board(height,width,win_length,pieces) 
    
    def get_board_size(self):
        return (self.board.height, self.board.width)

    def get_action_size(self):
        return self.board.width

    def get_next_state(self,player,action):
        return self.board.take_action(player,action)

    def get_symmetries(self):
        return self.board.pieces[::-1]

    def get_valid_actions(self):
        return gva(self.board.pieces)
        
    def get_winstate(self):
        return gws(self.board.pieces)

    def display_self(self):
        self.display(self.board.pieces)

    def __str__(self):
        return self.board.__str__()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(board)
        print(" -----------------------")
