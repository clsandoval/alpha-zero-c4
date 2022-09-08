#%%
import numpy as np
import math
import threading 
import time,threading,queue
import concurrent.futures
from network.nnet import nnet
from connect4.board import board
from connect4.game import Connect4
from mcts import MCTS as seq_mcts

class MCTS():
    
    def __init__(self,game,nnet, num_sims = 500,cpuct =  1.0,temp=1):
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
        self.queue = queue.Queue()
        self.cv = threading.Condition()
        self.done_threads = 0
        self.state_cache = {}
        self.locks = {}


    def network_thread(self):
        """
        Worker thread batching requests to network
        """
        while True:
            if self.done_threads == self.num_sims:
                return
            length = self.queue.qsize()
            samples = [self.queue.get() for i in range(length)]
            states = [i[0] for i in samples]
            batch = np.array([i[1] for i in samples])
            print(batch.shape)
            preds = self.nnet.net(batch)
            for i in range(len(states)):
                self.state_cache[states[i]] = (preds[0][i],preds[1][i])
            with self.cv:
                self.cv.notify_all()

    def get_probs(self, state):
        """
        Returns action probabilities of length (action_size)
        """
        #Store original board config
        state_str = str(state)
        self.done_threads=0
        threads = [threading.Thread(target=self.network_thread)]
        for i in range(self.num_sims):
            threads.append(threading.Thread(target=self.search, args=[state.copy(),True]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

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
        

    def search(self,state, root =False):
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
            if root == True:
                self.done_threads+=1
            return -player
        if not any(valid_actions):
            if root == True:
                self.done_threads+=1
            return -1

        #Expand if leaf node
        #lock this mutex
        if state_str not in self.P:
            if state_str in self.locks:
                lock = self.locks[state_str]
            else:
                lock = threading.Lock()
                self.locks[state_str] = lock
            if lock.locked() == False:
                lock.acquire()
               
                if state_str in self.state_cache:
                    pi, value = self.state_cache[state_str]
                    self.P[state_str]= pi * valid_actions
                    self.N[state_str] = 0
                    if root == True:
                        self.done_threads+=1
                    lock.release()
                    return -value
                self.queue.put((state_str,state))
                with self.cv:
                    while state_str not in self.state_cache:
                        self.cv.wait()
                pi, value = self.state_cache[state_str]
                self.P[state_str]= pi * valid_actions
                self.N[state_str] = 0
                if root == True:
                    self.done_threads+=1
                lock.release()
                return -value
            
        
        #spin lock
        while state_str not in self.P:
            with self.cv:
                self.cv.wait()

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
        if (state_str,best_a) not in self.Q:
            self.Q[(state_str,best_a)] = -1
            self.Nsa[(state_str,best_a)] = 1 
        else:
            self.Q[(state_str,best_a)] -= 1

        next_state = self.game.board.take_action(1,best_a,state)
        value = self.search(next_state)

        #backup
        if (state_str,best_a) not in self.Q:
            self.Q[(state_str,best_a)] = value+1
            self.Nsa[(state_str,best_a)] = 1 
        else:
            self.Nsa[(state_str,best_a)] += 1
            self.Q[(state_str,best_a)] = (value+1 + self.Nsa[(state_str,best_a)] * self.Q[(state_str,best_a)]) / (self.Nsa[(state_str,best_a)] + 1) 
        self.N[state_str] += 1
        if root == True:
            self.done_threads+=1
        return -value
    

    # %%
if __name__ == "__main__":              
    start = time.perf_counter()
    cgame = Connect4(6,7,4)
    net = nnet((6,7),4,128,3)
    m = MCTS(cgame,net,num_sims=500)
    next_state = np.zeros((6, 7),dtype=np.int)
    m.get_probs(next_state)
    print(time.perf_counter()-start)
    # %%
    for i in range(100):
        start = time.perf_counter()
        m.get_probs(next_state)
        print(time.perf_counter()-start)

    # %%
