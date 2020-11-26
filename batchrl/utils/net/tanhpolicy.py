import abc
import torch
import numpy as np

from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal
from tianshou.data import Batch
from tianshou.data import to_torch

#from batchrl.utils.net.continuous import ActorProb
from tianshou.utils.net.continuous import ActorProb



class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size(), device=self.normal_mean.device),
                torch.ones(self.normal_std.size(), device=self.normal_mean.device)
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
        

class TanhGaussianPolicy(ActorProb):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -5
    MEAN_MIN = -9.0
    MEAN_MAX = 9.0
    
    
    def get_actions(self, obs_np, deterministic=False):
        return self(obs_np, deterministic=deterministic)[0]
    
    def atanh(self,x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5*torch.log(one_plus_x/ one_minus_x)

    def log_prob(self, obs, actions):
        obs = to_torch(obs, device=self.device, dtype=torch.float32)
        actions = to_torch(actions, device=self.device, dtype=torch.float32)
        raw_actions = self.atanh(actions)
        logits, h = self.preprocess(obs)
        
        mean = self.mu(logits)
        mean = torch.clamp(mean, self.MEAN_MIN, self.MEAN_MAX)
        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(mu))
            std = log_std.exp()

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs,
            state=None,
            infor={},
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        s = to_torch(obs, device=self.device, dtype=torch.float32)
        logits, h = self.preprocess(s, state)
        mean = self.mu(logits)
        
        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(mu))
            std = log_std.exp()

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class MakeDeterministic(nn.Module):
    def __init__(self, stochastic_policy,device,process_fn=torch.nn.Identity):
        super().__init__()
        self.stochastic_policy = stochastic_policy
        self.process_fn = process_fn
        self.device = device

    def forward(self, batch, state=None, info ={}):
        observation = to_torch(batch.obs,torch.float,self.device)
        act =  self.stochastic_policy.get_actions(observation, deterministic=True)
        res = Batch(logits=None, act=act, state=state, dist=None, log_prob=None)
        
        return res