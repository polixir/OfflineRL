import torch
from batchrl.utils.exp import select_free_cuda

task = "Walker2d-v3"
dataset_path = "/home/revive/syg/datasets/revive/walker/Walker2d-v3-low-999-train.npz"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

actor_features = 256
actor_layers = 2

batch_size = 256
steps_per_epoch = 1000
max_epoch = 250

actor_lr = 1e-3

#tune
params_tune = {
    "actor_lr" : {"type" : "continuous", "value": [1e-4, 1e-3]},
}

#tune
grid_tune = {
    "actor_lr" : [1e-3, 1e-5],
    "actor_features" : [64, 1024],
    "actor_layers" : [1, 5],
}
