import torch
import gym

steps_per_epoch = 1000
max_epoch = 100
batch_size = 256
state_dim = None
action_dim = None
device = 'cpu'
alpha = 2.5
beta = 2.0
k = 1
policy_freq = 2
noise_clip = 0.5
policy_noise = 2
discount = 0.99
tau = 0.005
expl_noise = 0.1
critic_lr = 3e-4
actor_lr = 3e-4
max_action = 1.0