import numpy as np 
import sys 
from board import board
sys.path.append('..')


class Connect4():
    """ 
    Connect4 class implementing the connect4 game engine
    """

    def __init__(self,height, width, win_length,pieces=None):
        self.board = board(height,width,win_length,pieces) 
    
    def get_board_size():
        pass

    def get_action_size():
        pass

    def get_next_state():
        pass

    def get_valid_actions():
        pass

    def get_symmetries():
        pass

    def change_turns():
        pass
