import numpy as np 
import sys 
sys.path.append('..')


def get_row(pieces,column):
    for i in range(len(pieces)-1,-1,-1):
        if pieces[i][column] == 0:
            return i  
    return -1

def check_winner_straight(pieces,wlength=4):
    for i in range(len(pieces)):
        for j in range(len(pieces[0])-wlength+1):
            if all(pieces[i][j:j+wlength]): return True
    return False



def check_winner_diagonal(pieces,wlength=4):
    pass

def get_valid_actions(pieces):
    return pieces[0] == 0

def get_winstate(pieces):
    #check if current player wins
    for player in [1,-1]:
        player_pieces = pieces == player
        if check_winner_straight(player_pieces) or check_winner_straight(player_pieces.transpose()) or check_winner_diagonal(player_pieces):
            return (True,player) 
        

    return (False,1)