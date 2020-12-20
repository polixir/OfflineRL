import torch
from batchrl.utils.exp import select_free_cuda

task = "Walker2d-v3"
dataset_path = "/home/revive/syg/datasets/revive/walker/Walker2d-v3-low-999-train.npz"
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

learnable_alpha = False
transition_lr = 3e-4
actor_lr = 3e-4
critic_lr = 3e-4
target_entropy = -3
discount = 0.99
soft_target_tau = 5e-3

horizon = 5
lam = 5

#tune
params_tune = {
    "transition_lr" : {"type" : "discrete", "value": [1e-4, 3e-4]},
    "actor_lr" : {"type" : "discrete", "value": [1e-4, 3e-4]},
    "critic_lr" : {"type" : "discrete", "value": [1e-4, 3e-4]},
    "buffer_size" : {"type" : "discrete", "value": [5e5, 1e6, 2e6]},
    "real_data_ratio" : {"type" : "discrete", "value": [0, 0.05, 0.1]},
    "horzion" : {"type" : "discrete", "value": [1, 2, 5]},
    "lam" : {"type" : "continuous", "value": [1, 10]},
    "batch_size" : {"type": "discrete", "value": [128, 256, 512]},
    "learnable_alpha" : {"type" : "discrete", "value": [True, False]},
}

#tune
grid_tune = {
    "buffer_size" : [5e5, 1e6, 2e6],
    "real_data_ratio" : [0, 0.05, 0.1],
    "horzion" : [1, 2, 5],
    "lam" : [1, 5, 10],
    "batch_size" : [128, 256, 512],
    "learnable_alpha" : [True, False],
}