#%%
from coach import Coach
from network.nnet import nnet 
import wandb
wandb.init(project = "alphazero-c4-multithreaded")
net = nnet((6,7),4,128,3)
#%%
print("network initialized")
coach = Coach(net,100,num_sims=500,board_shape = (6,7), win_length=4,iterations=1000)
coach.train()

# %%

# %%
