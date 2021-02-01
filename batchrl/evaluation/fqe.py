# https://arxiv.org/abs/2007.09055
# Hyperparameter Selection for Offline Reinforcement Learning
from copy import deepcopy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from functools import partial
import pickle
import math
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from tianshou.data import to_numpy, to_torch

from batchrl.utils.net.common import MLP
from batchrl.utils.net.continuous import DistributionalCritic

class FQE:
    # https://arxiv.org/abs/2007.09055
    # Hyperparameter Selection for Offline Reinforcement Learning
    def __init__(self,
                 policy, 
                 buffer,
                 q_hidden_features=1024,
                 q_hidden_layers=4,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.policy = deepcopy(policy)
        self.buffer = buffer
        self.critic_hidden_features = q_hidden_features
        self.critic_hidden_layers = q_hidden_layers
        self._device = device

    def train_estimator(self,
                        init_critic=None, 
                        discount=0.99,
                        target_update_period=100,
                        critic_lr=1e-4,
                        num_steps=250000,
                        polyak=0.0,
                        batch_size=256):

        min_reward = self.buffer.rew.min()
        max_reward = self.buffer.rew.max()

        # enlarge the interval by 40%
        max_value = (1.2 * max_reward - 0.2 * min_reward) / (1 - discount)
        min_value = (1.2 * min_reward - 0.2 * max_reward) / (1 - discount)

        data = self.buffer.sample(batch_size)
        input_dim = data.obs.shape[-1] + data.act.shape[-1]
        critic = MLP(input_dim, 1, self.critic_hidden_features, self.critic_hidden_layers).to(self._device)
        if init_critic is not None: critic.load_state_dict(init_critic.state_dict())
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        target_critic = deepcopy(critic).to(self._device)
        target_critic.requires_grad_(False)

        print('Training Fqe...')
        for t in range(num_steps):
            data = self.buffer.sample(batch_size)
            data = to_torch(data, torch.float32, device=self._device)
            r = data.rew
            terminals = data.done
            o1 = data.obs
            a1 = data.act

            o2 = data.obs_next
            a2 = self.policy.get_action(o2)
            q_target = target_critic(torch.cat((o2, a2), -1)).detach()
            current_discount = discount * (1 - terminals)
            backup = r + current_discount * q_target
            backup = torch.clamp(backup, min_value, max_value) # prevent explosion
            
            q = critic(torch.cat((o1, a1), -1))
            critic_loss = ((q - backup) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        
            if t % target_update_period == 0:
                with torch.no_grad():
                    for p, p_targ in zip(critic.parameters(), target_critic.parameters()):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
        return critic

def fqe_eval_fn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def fqe_eval(policy, buffer, start_index):
        policy = deepcopy(policy)
        policy = policy.to(device)

        Fqe = FQE(policy, buffer,
                  q_hidden_features=1024,
                  q_hidden_layers=4,
                  device=device)

        critic = Fqe.train_estimator(discount=0.99,
                                     target_update_period=100,
                                     critic_lr=1e-4,
                                     num_steps=250000)

        data = buffer[start_index]
        obs = data.obs
        obs = torch.tensor(obs).float()
        estimate_q0 = []
        with torch.no_grad():
            for o in torch.split(obs, 256, dim=0):
                o = o.to(device)
                a = policy.get_action(o)
                init_sa = torch.cat((o, a), -1).to(device)
                estimate_q0.append(critic(init_sa).cpu())
        estimate_q0 = torch.cat(estimate_q0, dim=0)
        res = OrderedDict()
        res["Estimate_q0"] = estimate_q0.mean().item()
        return res
    return fqe_eval