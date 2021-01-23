import torch
from batchrl.utils.exp import select_free_cuda

#task = "Walker2d-v3"
task = "Hopper-v3"
#dataset_path = "/home/revive/syg/datasets/revive/walker/Walker2d-v3-low-999-train.npz"
dataset_path = "/home/revive/syg/datasets/revive/hopper/Hopper-v3-low-999-train.npz"
dataset_dir = "/home/revive/syg/datasets/revive//walker2d"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

max_epoch = 300
steps_per_epoch = 1000
policy_bc_steps = 40000

batch_size = 256
hidden_layer_size = 256
layer_num = 5
actor_lr=1E-4
critic_lr=3E-4
reward_scale=1
use_automatic_entropy_tuning=True
target_entropy = None
discount = 0.99
soft_target_tau=5e-3

# min Q
explore=1.0
temp=1.0
min_q_version=3
min_q_weight=1.0

# lagrange
with_lagrange=False
lagrange_thresh=10.0

# extra params
num_random=10
type_q_backup= "none"
q_backup_lmbda = 0.75
deterministic_backup=False

discrete = False

#tune
params_tune = {
    "actor_lr" : {"type" : "discrete", "value":[1e-4, 3e-4]},
    "min_q_version" : {"type" : "discrete", "value":[2, 3]},
    "min_q_weight" : {"type": "discrete", "value":[5, 10]},
    "lagrange_thresh" : {"type": "discrete", "value":[-1, 2, 5, 10]},
    "type_q_backup" : {"type": "discrete", "value":["max", "none"]},
}

#tune
grid_tune = {
    #"actor_lr" : [1e-4, 3e-4],
    "min_q_version" : [2, 3],
    "min_q_weight" : [5, 10],
    "lagrange_thresh" : [-1, 2, 5, 10],
    #"type_q_backup" : ["max", "none"],
}
