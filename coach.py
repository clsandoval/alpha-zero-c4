#%%
import numpy as np
import random
import time
import os
import wandb
from tqdm import tqdm
from mcts import MCTS
from network.nnet import nnet
from connect4.game import Connect4
from collections import deque
from arena import Arena
from players import NetworkPlayer, RandomPlayer
#%%
class Coach():
    "Implements an environment where agents learn through self play"
    
    def __init__(self,nnet, num_eps,board_shape = (5,5),win_length=4, num_sims = 25, iterations = 10):
        self.board_shape = board_shape
        self.win_length = win_length
        self.nnet=nnet 
        self.num_eps = num_eps
        self.num_sims = 25
        self.examples = []
        self.iterations = iterations

    def episode(self):
        self.game = Connect4(self.board_shape[0],self.board_shape[1],self.win_length)
        self.mcts = MCTS(self.game,self.nnet, num_sims = self.num_sims)
        states = []
        problist = []
        playerlist = []
        player_ctr = 1
        while True:
            probs = self.mcts.get_probs()
            actions = np.arange(len(probs))
            action = np.random.choice(actions,p=probs)
            self.game.get_next_state(1,action)
            states.append(self.game.board.pieces)
            states.append(np.fliplr(self.game.board.pieces))
            problist.append(probs)
            problist.append(probs)
            playerlist.append(player_ctr)
            playerlist.append(player_ctr)
            win, player = self.game.get_winstate()
            valid_actions = self.game.get_valid_actions()
            if win or not any(valid_actions):
                return [(states[i],problist[i], playerlist[i] * -player) for i in range(len(states))]
            player_ctr *= -1

    def train(self):
        for it in (range(self.iterations)):
            print("Starting Iteration")
            for i in (range(self.num_eps)):
                start = time.perf_counter()
                result = self.episode()
                self.examples = self.examples + result
                ep_time = time.perf_counter()-start
                wandb.log({'episode_time': ep_time})
            wandb.log({'examples_in_mem':len(self.examples)})
            if len(self.examples) > 200000:
                excess = len(self.examples) - 200000
                self.examples = self.examples[excess:]
            random.shuffle(self.examples)
            self.nnet.train(self.examples)
            current_network = self.nnet
            best_network = self.nnet
            if os.path.exists("models/current_model"):
                print("Loading Model")
                best_network.net = self.nnet.load_checkpoint()

            print("Comparing Networks")
            random_player = RandomPlayer()
            current_player = NetworkPlayer(current_network,name="new network")
            current_player_vs_random = NetworkPlayer(current_network,name="cvr network")
            best_player = NetworkPlayer(best_network,name="best network")
            arena = Arena(best_player,current_player,self.board_shape[0],self.board_shape[1],self.win_length,log=True)
            res = arena.pit()
            if res == False: #player 1 wins
                print("player 1 wins, retaining old network")
                best_player.nnet.save_checkpoint()
            else: #player 2 wins
                print("player 2 wins, keeping new network")
                self.nnet.save_checkpoint()
            random_arena = Arena(random_player,current_player_vs_random,self.board_shape[0],self.board_shape[1],self.win_length,log=True,battles=20)
            random_arena.pit()

            

# %%

# %%
