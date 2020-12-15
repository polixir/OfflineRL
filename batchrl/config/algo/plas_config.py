import torch
from batchrl.utils.exp import select_free_cuda

task = "ib"
dataset_path = "/home/revive/syg/datasets/revive/ib/ib-low-99-train.npz"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

aim_path = "/home/revive/syg/polixir/BatchRL/examples"

vae_iterations = 500000
vae_hidden_size = 750
vae_batch_size = 100
vae_kl_weight = 0.5
#vae_pretrain_model = "/tmp/vae_499999.pkl"


latent = False
layer_num = 5
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

#tune
params_tune = {
    "vae_iterations" : {"type" : "discrete", "value":[50000, 100000, 500000,]},
    "actor_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "vae_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "actor_batch_size" : {"type": "discrete", "value":[128, 256, 512]},
    "latent" : {"type": "discrete", "value":[True, False]},
    "lmbda" :{"type": "discrete", "value":[0.65, 0.75, 0.85]},
}

#tune
grid_tune = {
    "vae_iterations" : [50000, 100000, 500000],
    "actor_batch_size" : [128, 256],
    "latent" : [True, False],
    "lmbda" : [0.65, 0.75, 0.85]
}
