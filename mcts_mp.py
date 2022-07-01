#%%
import numpy as np
import math
import time,threading
from network.nnet import nnet
from connect4.board import board
from connect4.game import Connect4

class MCTS():
    
    def __init__(self,game,nnet, num_sims = 25,cpuct =  1.0,temp=1):
        self.num_sims = num_sims
        self.game = game
        self.nnet = nnet
        self.dp_w= {}
        self.Q = {}
        self.Nsa = {}
        self.N = {}
        self.P = {}
        self.cpuct = cpuct
        self.temp = temp
        #thread stuff
        self.queue = threading.Queue()
        self.cv = threading.Condition()
        self.done_threads = 0
        self.state_cache = {}


    def network_thread(self):
        """
        Worker thread batching requests to network
        """
        if self.done_threads == self.num_sims:
            return
        samples = [i.get() for i in self.queue]
        states = [i[0] for i in samples]
        batch = np.array([i[1] for i in samples])
        preds = self.nnet.net(batch)
        for i in range(states):
            self.cache[states[i]] = (preds[0][i],preds[1][i])
        with self.cv:
            self.cv.notify_all()

    def get_probs(self, state):
        """
        Returns action probabilities of length (action_size)
        """
        #Store original board config
        state_str = str(state)
        for i in range(self.num_sims):
            self.search(state.copy())

        #return exponentiated counts of state actions visited
        counts = [self.Nsa[(state_str,a)] if (state_str,a) in self.Nsa else 0 for a in range(self.game.get_action_size())]
        
        if self.temp ==0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs 

        counts = [x ** (1./self.temp) for x in counts]
        count_sum = float(sum(counts))
        probs = [x/count_sum for x in counts]
        return probs
        

    def search(self,state):
        """ 
        Execute one monte carlo tree search returning the value at the given state
        """
        state_str = str(state)
        
        #Check if winner, return -1 if so
        if state_str in self.dp_w:
            win,player,valid_actions = self.dp_w[state_str]
        else:
            valid_actions = self.game.get_valid_actions(state)
            win, player = self.game.get_winstate(state)
        if win:
            return -player
        if not any(valid_actions):
            return -1

        #Expand if leaf node
        if state_str not in self.P:
            if state_str in self.state_cache:
                pi, value = self.state_cache[state_str]
                self.P[state_str]= pi * valid_actions
                self.N[state_str] = 0
                return -value
            self.queue.put((state_str,state))
            with self.cv:
                while state_str not in self.state_cache:
                    self.cv.wait()
            pi, value = self.state_cache[state_str]
            self.P[state_str]= pi * valid_actions
            self.N[state_str] = 0
            return -value
            

        #Evaluate edge
        max_q_u, best_a = -float("inf"), -1
        valid_moves = [i for i, x in enumerate(valid_actions) if x]
        for action in valid_moves:
            state_action = (state_str,action)
            if state_action not in self.Q:
                q_u = self.cpuct * self.P[state_str][action] * math.sqrt(self.N[state_str] + .00000001) 
            else: 
                u = self.cpuct * self.P[state_str][action] * math.sqrt(self.N[state_str])/(1+self.Nsa[(state_str,action)]) 
                q_u = self.Q[state_action] + u
            if q_u > max_q_u:
                max_q_u = q_u
                best_a = action
        #take action and pass to next player
        next_state = self.game.board.take_action(1,best_a,state)
        value = self.search(next_state)

        #backup
        if (state_str,best_a) not in self.Q:
            self.Q[(state_str,best_a)] = value
            self.Nsa[(state_str,best_a)] = 1 
        else:
            self.Nsa[(state_str,best_a)] += 1
            self.Q[(state_str,best_a)] = (value + self.Nsa[(state_str,best_a)] * self.Q[(state_str,best_a)]) / (self.Nsa[(state_str,best_a)] + 1) 
        self.N[state_str] += 1
        return -value
    
                    
# %%

# %%
