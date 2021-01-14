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
from tianshou.data import to_numpy,to_torch

ACTIVATION_CREATORS = {
    'relu' : lambda dim: nn.ReLU(inplace=True),
    'elu' : lambda dim: nn.ELU(),
    'leakyrelu' : lambda dim: nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'tanh' : lambda dim: nn.Tanh(),
    'sigmoid' : lambda dim: nn.Sigmoid(),
    'identity' : lambda dim: nn.Identity(),
    'prelu' : lambda dim: nn.PReLU(dim),
    'gelu' : lambda dim: nn.GELU(),
}

def get_models_parameters(*models):
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    return parameters

class MLP(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, hidden_layers,
                 norm=None, hidden_activation='leakyrelu', output_activation='identity'):
        super(MLP, self).__init__()

        hidden_activation_creator = ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creator(out_features)
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
                if norm:
                    if norm == 'ln':
                        net.append(nn.LayerNorm(hidden_features))
                    elif norm == 'bn':
                        net.append(nn.BatchNorm1d(hidden_features))
                    else:
                        raise NotImplementedError(f'{norm} does not supported!')
                net.append(hidden_activation_creator(hidden_features))
            net.append(nn.Linear(hidden_features, out_features))
            net.append(output_activation_creator(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out

class FQE:
    # https://arxiv.org/abs/2007.09055
    # Hyperparameter Selection for Offline Reinforcement Learning
    def __init__(self,
                 policy, buffer,
                 q_hidden_features=None,
                 q_hidden_layers=None,
                 ):
        self.policy = policy
        self.buffer = buffer
        self.critic_hidden_features = q_hidden_features
        self.critic_hidden_layers = q_hidden_layers
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_estimator(self, discount=0.99,
                         target_update_period=100,
                         critic_lr=1e-5,
                         num_steps=10000,
                         polyak=0.95,
                         batch_size=256):
        #writer = SummaryWriter('fqe')

        batch = self.buffer.sample(batch_size)
        data = to_torch(batch, torch.float, device=self._device)
        input_dim = data.obs.shape[-1] + data.act.shape[-1]
        critic = MLP(input_dim, 1, self.critic_hidden_features, self.critic_hidden_layers).to(self._device)
        critic_optimizer = torch.optim.Adam([{'params': critic.parameters(), 'lr':critic_lr }])
        target_critic = deepcopy(critic).to(self._device)

        for p in target_critic.parameters():
            p.requires_grad = False
        print('Training Fqe...')
        for t in tqdm(range(num_steps)):
            batch = self.buffer.sample(batch_size)
            data = to_torch(batch, torch.float, device=self._device)
            r = data.rew
            terminals = data.done
            o1 = data.obs
            a1 = data.act

            if terminals[0,0]:
                backup = r
            else:
                o2 = data.obs_next
                a2 = self.policy.get_action(o2)
                o2 = torch.tensor(o2).to(self._device)
                a2 = torch.tensor(a2).to(self._device)
                q_target = target_critic(torch.cat((o2, a2), -1)).detach()
                current_discount = discount
                backup = r + current_discount * q_target
                backup = torch.clamp(backup, 0, 600)
            q = critic(torch.cat((o1, a1), -1))
            critic_loss = ((q - backup) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(get_models_parameters(critic), 0.5)
            critic_optimizer.step()
        
            #writer.add_scalar('critic_loss', critic_loss.item(), t)
            if t % target_update_period == 0:
                with torch.no_grad():
                    for p, p_targ in zip(critic.parameters(), target_critic.parameters()):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
        #writer.close()
        return critic

def fqe_eval_fn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def fqe_eval(policy, buffer):
        Fqe = FQE(policy, buffer,
                  q_hidden_features=1024,
                  q_hidden_layers=4)

        critic = Fqe.train_estimator(discount=0.99,
                                     target_update_period=100,
                                     critic_lr=1e-4,
                                     num_steps=250000,
                                     polyak=0)

        eval_size = 20000
        batch = buffer[:eval_size]
        data = to_torch(batch, torch.float)
        o0, a0 = data.obs, data.act
        init_sa = torch.cat((o0,a0), -1).to(device)
        estimate_q0 = critic(init_sa)
        res = OrderedDict()
        res["Estimate_q0"] = estimate_q0.mean().item()
        return res
    return fqe_eval