#%%
from coach import Coach
from network.nnet import nnet 
from arena import Arena
from players import RandomPlayer, NetworkPlayer, HumanPlayer
#%%
net = nnet((6,7),4,128,3)
net.net = net.load_checkpoint()
p1 = NetworkPlayer(net,name = "network")
p2 = HumanPlayer(name="new")
p3 = RandomPlayer()
arena = Arena(p1,p2,6,7,4,battles=6)
res = arena.pit(verbose=True)
# %%
