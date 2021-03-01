import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42 

device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 400
hidden_layers = 2
transition_layers = 4

transition_init_num = 7
transition_select_num = 5

real_data_ratio = 0.5

transition_batch_size = 256
policy_batch_size = 256
data_collection_per_epoch = 50e3
buffer_size = 250e3
steps_per_epoch = 1000
max_epoch = 300

learnable_alpha = False
transition_lr = 1e-3
actor_lr = 1e-4
critic_lr = 3e-4
target_entropy = -6
discount = 0.99
soft_target_tau = 1e-2

num_samples = 10
learnable_beta = False
base_beta = 0.5
lagrange_thresh = 5
with_important_sampling = True

horizon = 5

#tune
params_tune = {
    "buffer_size" : {"type" : "discrete", "value": [1e6, 2e6]},
    "real_data_ratio" : {"type" : "discrete", "value": [0.05, 0.1, 0.2]},
    "horzion" : {"type" : "discrete", "value": [1, 2, 5]},
    "lam" : {"type" : "continuous", "value": [0.1, 10]},
    "learnable_alpha" : {"type" : "discrete", "value": [True, False]},
}

#tune
grid_tune = {
    "horizon" : [1, 5],
    "with_important_sampling" : [True, False],
    "base_beta" : [0.5, 1, 5],
    "real_data_ratio" : [0.5, 0.75],
}
