import numpy as np 
from mcts import MCTS
from connect4.game import Connect4
import wandb


class Arena():
    """Implements an arena where players are pitted against each other to ensure that only the best network is retained"""

    def __init__(self,player1,player2,height,width,win_length,battles=40,log=False):
        self.battles = battles 
        self.player1 = player1
        self.player2 = player2
        self.height = height
        self.width = width 
        self.log =log
        self.win_length = win_length
        self.wins=[0,0]

    def battle(self,verbose):
        """Pits the current and candidate networks against each other and returns the network with more wins"""
        candidate_wins,best_wins = 0,0
        for i in range(self.battles):
            game = Connect4(self.height,self.width,self.win_length)
            players = [self.player1,self.player2]
            for p in players:
                p.init_game(game)
            player_ctr = 0
            while True:
                valid_actions = game.get_valid_actions()
                action = players[player_ctr].decision(game.board,valid_actions)
                game.get_next_state(1,action)
                valid_actions_2 = game.get_valid_actions()
                win, player = game.get_winstate()
                if player == -1:
                    self.wins[player_ctr] += 1
                    if verbose:
                        print("player {} wins".format(player_ctr))
                    break
                if not any(valid_actions_2):
                    break
                player_ctr = (player_ctr + 1) %2

    def pit(self,verbose = False):
        self.battle(verbose)
        if self.log:
            wandb.log({self.player1.name: self.wins[0], self.player2.name: self.wins[1]})
        print("Candidate: {} Current: {}".format(self.wins[0],self.wins[1]))
        if self.wins[0]/(self.battles) > .55:
            return 0
        return 1
        


                

