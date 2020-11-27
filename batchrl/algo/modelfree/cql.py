# Conservative Q-Learning for Offline Reinforcement Learning
# https://arxiv.org/abs/2006.04779
# https://github.com/aviralkumar2907/CQL
import abc
import copy
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from torch import optim
from tianshou.data import to_torch

from batchrl.utils.env import get_env_shape
from batchrl.utils.net.common import Net
from batchrl.utils.net.continuous import Critic
from batchrl.utils.net.tanhpolicy import TanhGaussianPolicy


def algo_init(args):
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    else:
        obs_shape, action_shape = get_env_shape(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    
    net_a = Net(layer_num = args['layer_num'], 
                     state_shape = obs_shape, 
                     device = args['device'],
                     hidden_layer_size = args['hidden_layer_size'])
    
    actor = TanhGaussianPolicy(preprocess_net = net_a,
                                action_shape = action_shape,
                                device = args['device'],
                                hidden_layer_size = args['hidden_layer_size'],
                                conditioned_sigma = True,
                              ).to(args['device'])
    
    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])
    
    net_c1 = Net(layer_num = args['layer_num'],
                  state_shape = obs_shape,  
                  action_shape = action_shape,
                  concat = True, 
                  device = args['device'],
                  hidden_layer_size = args['hidden_layer_size'])
    critic1 = Critic(preprocess_net = net_c1, 
                     device = args['device'], 
                     hidden_layer_size = args['hidden_layer_size'],
                    ).to(args['device'])
    critic1_optim = optim.Adam(critic1.parameters(), lr=args['critic_lr'])
    
    net_c2 = Net(layer_num = args['layer_num'],
                  state_shape = obs_shape,  
                  action_shape = action_shape,
                  concat = True, 
                  device = args['device'],
                  hidden_layer_size = args['hidden_layer_size'])
    critic2 = Critic(preprocess_net = net_c2, 
                     device = args['device'], 
                     hidden_layer_size = args['hidden_layer_size'],
                    ).to(args['device'])
    critic2_optim = optim.Adam(critic2.parameters(), lr=args['critic_lr'])
    """
    critic1_target = copy.deepcopy(critic1)
    critic2_target = copy.deepcopy(critic2)
    
    if args["auto_alpha"]:
        if args["target_entropy"]:
            target_entropy = args["target_entropy"]
        else:
            target_entropy = -np.prod(action_shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
            alpha_optim = optim.Adam([log_alpha], lr=arg['alpha_lr'])
            alpha = (target_entropy, log_alpha, alpha_optim)
    """ 
    return {
        "actor" : {"net" : actor, "opt" : actor_optim},
        "critic1" : {"net" : critic1, "opt" : critic1_optim},
        "critic2" : {"net" : critic2, "opt" : critic2_optim},
    }


class AlgoTrainer():
    def __init__(self, algo_init, args):
        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]
        
        self.critic1 = algo_init["critic1"]["net"]
        self.critic1_opt = algo_init["critic1"]["opt"]
        self.critic2 = algo_init["critic2"]["net"]
        self.critic2_opt = algo_init["critic2"]["opt"]
        
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.device = args["device"]
        self.soft_target_tau = args["soft_target_tau"]
        self.use_automatic_entropy_tuning = args["use_automatic_entropy_tuning"]
        
        if self.use_automatic_entropy_tuning:
            if args["target_entropy"]:
                self.target_entropy = args["target_entropy"]
            else:
                self.target_entropy = -np.prod(args["action_shape"]).item() 
            self.log_alpha = torch.zeros(1,requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha],
                lr=args["actor_lr"],
            )
        
        self.with_lagrange = args["with_lagrange"]
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = torch.zeros(1,requires_grad=True, device=self.device)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=args["critic_lr"],
            )

        self.critic_criterion = nn.MSELoss()

        self.discount = args["discount"]
        self.reward_scale = args["reward_scale"]
        self.policy_bc_steps = args["policy_bc_steps"]
        
        ## min Q
        self.temp = args["temp"]
        self.min_q_version = args["min_q_version"]
        self.min_q_weight = args["min_q_weight"]


        self.max_q_backup = args["max_q_backup"]
        self.deterministic_backup = args["deterministic_backup"]
        self.num_random = args["num_random"]

        # For implementation on the 
        self.discrete = args["discrete"]
        
        self._n_train_steps_total = 0
        self._current_epoch = 0
        
    def sync_weight(self) -> None:
        for o, n in zip(
            self.critic1_target.parameters(), self.critic1.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self.soft_target_tau) + n.data * self.soft_target_tau)
        for o, n in zip(
            self.critic2_target.parameters(), self.critic2.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self.soft_target_tau) + n.data * self.soft_target_tau)
    
    def _get_tensor_values(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp, reparameterize=True, return_log_prob=True,
        )
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions
        
    def train(self, batch):
        self._current_epoch += 1
        batch = to_torch(batch, torch.float, device=self.device)
        rewards = batch.rew
        terminals = batch.done
        obs = batch.obs
        actions = batch.act
        next_obs = batch.obs_next

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.actor(
            obs, reparameterize=True, return_log_prob=True,
        )
        
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.critic1(obs, new_obs_actions),
            self.critic2(obs, new_obs_actions),
        )

        if self._current_epoch < self.policy_bc_steps:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.actor.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()
        else:
            policy_loss = (alpha*log_pi - q_new_actions).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()
        
        """
        QF Loss
        """
        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)
        
        new_next_actions, _, _, new_log_pi, *_ = self.actor(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.actor(
            obs, reparameterize=True, return_log_prob=True,
        )

        if not self.max_q_backup:
            target_q_values = torch.min(
                self.critic1_target(next_obs, new_next_actions),
                self.critic2_target(next_obs, new_next_actions),
            )
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values.detach()
            
        qf1_loss = self.critic_criterion(q1_pred, q_target)
        qf2_loss = self.critic_criterion(q2_pred, q_target)

        ## add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(self.device) #.cuda().detach()
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.actor)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.actor)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic2)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        """
        Update critic networks
        """
        self.critic1_opt.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        qf2_loss.backward()
        self.critic2_opt.step()

        """
        Soft Updates
        """
        self.sync_weight()
        self._n_train_steps_total += 1