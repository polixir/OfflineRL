import torch
from batchrl.utils.exp import select_free_cuda

#task = "Walker2d-v3"
task = "Hopper-v3"
dataset_path = "/home/revive/syg/datasets/revive/hopper/Hopper-v3-low-999-train.npz"
dataset_dir = "/home/revive/syg/datasets/revive//walker2d"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

aim_path = "/home/revive/syg/polixir/BatchRL/examples"

max_timesteps = 1e6
eval_freq = 1e3

optimizer_parameters = {
    "lr": 3e-4,
    }

BCQ_threshold = 0.3

discount = 0.99
tau = 0.005
polyak_target_update = True
target_update_frequency=1
start_timesteps = 1e3
initial_eps = 0.1
end_eps = 0.1
eps_decay_period = 1
eval_eps = 0.001
buffer_size = 1e6
batch_size = 256
train_freq = 1