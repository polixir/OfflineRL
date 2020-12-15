import torch
from batchrl.utils.exp import select_free_cuda

task = "walker2d-medium-v0"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 200
hidden_layers = 2
transition_layers = 4

transition_init_num = 7
transition_select_num = 5

real_data_ratio = 0.05

batch_size = 256
transition_steps_per_epoch = 250
data_collection_per_epoch = 50e3
buffer_size = 1e6
steps_per_epoch = 1000
max_epoch = 200

transition_lr = 3e-4
actor_lr = 3e-4
critic_lr = 3e-4
target_entropy = -3
discount = 0.99
soft_target_tau = 5e-3

horizon = 5
lam = 5
