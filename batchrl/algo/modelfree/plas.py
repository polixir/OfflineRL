# PLAS: Latent Action Space for Offline Reinforcement Learning
# https://sites.google.com/view/latent-policy
# https://github.com/Wenxuan-Zhou/PLAS
import abc
import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import optim
from tianshou.data import to_torch

from batchrl.algo.base import BasePolicy
from batchrl.utils.env import get_env_shape, get_env_action_range
from batchrl.utils.net.common import Net
from batchrl.utils.net.continuous import Critic, Actor
from batchrl.utils.net.vae import VAE


def algo_init(args):
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    else:
        obs_shape, action_shape = get_env_shape(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
        
    max_action, _ = get_env_action_range(args["task"])
    vae = VAE(state_dim = obs_shape, 
              action_dim = action_shape, 
              latent_dim = action_shape*2, 
              max_action = max_action,
              device = args["device"],
              hidden_size=args["vae_hidden_size"]).to(args['device'])
    
    vae_opt = optim.Adam(vae.parameters(), lr=args["vae_lr"])
    
    net_a = Net(layer_num = args["layer_num"], 
                state_shape = obs_shape, 
                device = args["device"],
                hidden_layer_size = args["hidden_layer_size"])
    actor = Actor(preprocess_net = net_a,
                 action_shape = action_shape*2,
                 max_action = max_action,
                 device = args["device"],
                 hidden_layer_size = args["hidden_layer_size"]).to(args['device'])
    actor_opt = optim.Adam(actor.parameters(), lr=args["actor_lr"])
    
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
    critic1_opt = optim.Adam(critic1.parameters(), lr=args['critic_lr'])
    
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
    critic2_opt = optim.Adam(critic2.parameters(), lr=args['critic_lr'])
    
    return {
        "vae" : {"net" : vae, "opt" : vae_opt},
        "actor" : {"net" : actor, "opt" : actor_opt},
        "critic1" : {"net" : critic1, "opt" : critic1_opt},
        "critic2" : {"net" : critic2, "opt" : critic2_opt},
    }

class eval_policy():
    def __init__(self, vae, actor):
        self.vae = vae
        self.actor = actor

    def get_action(self, state):
        #state = to_torch(state, device=self.vae.device)
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.vae.device)
            action = self.vae.decode(state, z=self.actor(state)[0])
        return action.cpu().data.numpy().flatten()


class AlgoTrainer(BasePolicy):
    def __init__(self, algo_init, args):
        self.vae = algo_init["vae"]["net"]
        self.vae_opt = algo_init["vae"]["opt"]
        
        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]

        self.critic1 = algo_init["critic1"]["net"]
        self.critic1_opt = algo_init["critic1"]["opt"]
        
        self.critic2 = algo_init["critic2"]["net"]
        self.critic2_opt = algo_init["critic2"]["opt"]
        
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.args = args
        
    def _sync_weight(self, net_target, net, soft_target_tau = None) -> None:
        if soft_target_tau is None:
            soft_target_tau = self.args["soft_target_tau"]
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

        
    def _train_vae_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.args["device"])
        obs = batch.obs
        act = batch.act
        
        recon, mean, std = self.vae(obs, act)
        recon_loss = F.mse_loss(recon, act)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_opt.zero_grad()
        vae_loss.backward()
        self.vae_opt.step()
        
        return vae_loss.cpu().data.numpy(), recon_loss.cpu().data.numpy(), KL_loss.cpu().data.numpy()
        
    def _train_vae(self, replay_buffer):
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': []}
        for i in range(self.args["vae_iterations"]):
            batch = replay_buffer.sample(self.args["vae_batch_size"])
            vae_loss, recon_loss, KL_loss = self._train_vae_step(batch)
            logs['vae_loss'].append(vae_loss)
            logs['recon_loss'].append(recon_loss)
            logs['kl_loss'].append(KL_loss)
            if (i + 1) % 1000 == 0:
                print("VAE Epoch :", (i + 1) // 1000)
                print('Itr ' + str(i+1) + ' Training loss:' + '{:.4}'.format(vae_loss))

        
    def _train_policy(self, replay_buffer, eval_fn):
        for it in range(self.args["actor_iterations"]):
            batch = replay_buffer.sample(self.args["actor_batch_size"])
            batch = to_torch(batch, torch.float, device=self.args["device"])
            rew = batch.rew
            done = batch.done
            obs = batch.obs
            act = batch.act
            obs_next = batch.obs_next

            # Critic Training
            with torch.no_grad():
                action_next_actor,_ = self.actor_target(obs_next)
                action_next_vae = self.vae.decode(obs_next, z = action_next_actor)

                target_q1 = self.critic1_target(obs_next, action_next_vae)
                target_q2 = self.critic2_target(obs_next, action_next_vae)
 
                target_q = self.args["lmbda"] * torch.min(target_q1, target_q2) + (1 - self.args["lmbda"]) * torch.max(target_q1, target_q2)
                target_q = rew + (1 - done) * self.args["discount"] * target_q

            current_q1 = self.critic1(obs, act)
            current_q2 = self.critic2(obs, act)

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic1_opt.zero_grad()
            self.critic2_opt.zero_grad()
            critic_loss.backward()
            self.critic1_opt.step()
            self.critic2_opt.step()
            
            # Actor Training
            action_actor,_ = self.actor(obs)
            action_vae = self.vae.decode(obs, z = action_actor)
            actor_loss = -self.critic1(obs, act).mean()
            
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # update target network
            self._sync_weight(self.actor_target, self.actor)
            self._sync_weight(self.critic1_target, self.critic1)
            self._sync_weight(self.critic2_target, self.critic2)
            
            if (it + 1) % 1000 == 0:
                print("Policy Epoch :", (it + 1) // 1000)
                if eval_fn is None:
                    self.eval()
                else:
                    eval_fn(self.args["task"],eval_policy(self.vae, self.actor))
            
        
    def train(self, replay_buffer, eval_fn=None,):
        self._train_vae(replay_buffer)
        self._train_policy(replay_buffer, eval_fn)
        
        
