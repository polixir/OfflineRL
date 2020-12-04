import torch
from batchrl.utils.exp import select_free_cuda

task = "walker2d-medium-v0"
dataset_dir = "/home/revive/syg/datasets/walker2d/"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None

vae_iterations = 500000
vae_hidden_size = 750
vae_batch_size = 100

layer_num = 3
actor_batch_size = 100
hidden_layer_size = 256
actor_iterations = 500000
vae_lr = 1e-4
actor_lr = 1e-4
critic_lr = 1e-3
soft_target_tau = 0.005
lmbda = 0.75
discount = 0.99

max_latent_action = 2 
phi = 0.05