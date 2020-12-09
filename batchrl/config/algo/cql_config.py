import torch
from batchrl.utils.exp import select_free_cuda

task = "HalfCheetah-v3"
dataset_path = "/home/revive/syg/datasets/revive/HalfCheetah-v3-low-999-train.npz"
dataset_dir = "/home/revive/syg/datasets/walker2d/walker2d"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None


max_epoch = 1000
steps_per_epoch = 1000
policy_bc_steps = 40000

batch_size = 256
hidden_layer_size = 256
layer_num = 3
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
lagrange_thresh=0.0

# extra params
num_random=10
max_q_backup=False
deterministic_backup=False

discrete = False

#tune
params_tune = {
    "actor_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "layer_num" :{"type": "discrete", "value":[2, 3, 4]},
    "use_automatic_entropy_tuning" : {"type":"grid", "value":[True, False]},
}
