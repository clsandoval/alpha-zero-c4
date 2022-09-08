#%%
from network.nnet import nnet 
from arena import Arena
from connect4.game import Connect4
from players import RandomPlayer, NetworkPlayer, HumanPlayer
#%%
net = nnet((6,7),4,128,3)
net.net = net.load_checkpoint()
# %%
p1 = NetworkPlayer(net,name = "network")
p2 = HumanPlayer(name="new")
p3 = RandomPlayer()
arena = Arena(p1,p3,6,7,4,battles=6)
res = arena.pit(verbose=True)
# %%
import time 
import numpy as np
start = time.perf_counter()
game = Connect4(6,7,4)
net.net.predict(np.expand_dims(game.board.pieces,0))
time.perf_counter()-start
# %%
start = time.perf_counter()
game = Connect4(6,7,4)
b = np.array([game.board.pieces for i in range(32)])
preds = net.net.predict(b)
time.perf_counter()-start
# %%
