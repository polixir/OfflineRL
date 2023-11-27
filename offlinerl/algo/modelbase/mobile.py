# MOBILE: Model-Bellman Inconsistency Penalized Policy Optimization
# https://proceedings.mlr.press/v202/sun23q.html
# https://github.com/yihaosun1124/mobile

import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition
from offlinerl.utils.net.terminal_check import get_termination_fn

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape
        obs_shape, action_shape = get_env_shape(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'], args['transition_init_num']).to(args['device'])
    transition_optim = torch.optim.AdamW(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)
    terminal_fn = get_termination_fn(args['task'])

    net_a = Net(layer_num=args['hidden_layers'], 
                state_shape=obs_shape, 
                hidden_layer_size=args['hidden_layer_size'])

    actor = TanhGaussianPolicy(preprocess_net=net_a,
                               action_shape=action_shape,
                               hidden_layer_size=args['hidden_layer_size'],
                               conditioned_sigma=True).to(args['device'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["actor_lr"])
    
    critics = [
        MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
        for _ in range(args["num_q_ensemble"])
    ]
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args["critic_lr"])

    if args['lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args['max_epoch'])
    else:
        lr_scheduler = None

    return {
        "transition" : {"net" : transition, "opt" : transition_optim, "terminal_fn" : terminal_fn},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer},
        "critics" : {"net" : critics, "opt" : critics_optim},
        'lr_scheduler' : lr_scheduler
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.transition = algo_init['transition']['net']
        self.transition_optim = algo_init['transition']['opt']
        self.terminal_fn = algo_init['transition']['terminal_fn']
        self.selected_transitions = None

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']
        
        self.critics = algo_init['critics']['net']
        self.target_critics = deepcopy(self.critics)
        self.critic_optim = algo_init['critics']['opt']
        self.lr_scheduler = algo_init['lr_scheduler']

        self.device = args['device']

        self.args['buffer_size'] = int(self.args['data_collection_per_epoch']) * self.args['horizon'] * 5
        self.args['target_entropy'] = - self.args['action_shape']
        
    def train(self, train_buffer, val_buffer, callback_fn):
        if self.args['dynamics_path'] is not None:
            self.transition = torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.train_transition(train_buffer)
            if self.args['dynamics_save_path'] is not None: torch.save(self.transition, self.args['dynamics_save_path'])
        self.transition.requires_grad_(False)   
        self.train_policy(train_buffer, val_buffer, self.transition, callback_fn)
    
    def get_policy(self):
        return self.actor

    def train_transition(self, buffer):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['transition_batch_size']

        val_losses = [float('inf') for i in range(self.transition.ensemble_size)]

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                self._train_transition(self.transition, batch, self.transition_optim)
            new_val_losses = self._eval_transition(self.transition, valdata)
            print(new_val_losses)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                if new_loss < old_loss:
                    indexes.append(i)
                    val_losses[i] = new_loss

            if len(indexes) > 0:
                self.transition.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if cnt >= 5:
                break
        
        indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
        self.transition.set_select(indexes)
        return self.transition

    def train_policy(self, train_buffer, val_buffer, transition, callback_fn):
        real_batch_size = int(self.args['policy_batch_size'] * self.args['real_data_ratio'])
        model_batch_size = self.args['policy_batch_size']  - real_batch_size
        
        model_buffer = ModelBuffer(self.args['buffer_size'])

        obs_max = torch.as_tensor(train_buffer['obs'].max(axis=0)).to(self.device)
        obs_min = torch.as_tensor(train_buffer['obs'].min(axis=0)).to(self.device)
        rew_max = train_buffer['rew'].max()
        rew_min = train_buffer['rew'].min()

        for epoch in range(self.args['max_epoch']):
            # collect data
            with torch.no_grad():
                obs = train_buffer.sample(int(self.args['data_collection_per_epoch']))['obs']
                obs = torch.tensor(obs, device=self.device)
                for t in range(self.args['horizon']):
                    action = self.actor(obs).sample()
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = transition(obs_action)
                    next_obses = next_obs_dists.sample()
                    rewards = next_obses[:, :, -1:]
                    next_obses = next_obses[:, :, :-1]

                    # compute model-bellman inconcistency
                    next_obes_ = torch.stack([next_obs_dists.sample()[...,:-1] for i in range(10)], 0)
                    num_samples, num_ensembles, batch_size, obs_dim = next_obes_.shape
                    next_obes_ = next_obes_.reshape(-1, obs_dim)
                    next_actions_ = self.actor(next_obes_).sample()
                    next_obs_acts_ = torch.cat([next_obes_, next_actions_], dim=-1)
                    next_qs_ =  torch.cat([target_critic(next_obs_acts_) for target_critic in self.target_critics], 1)
                    next_qs_ = torch.min(next_qs_, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
                    uncertainty = next_qs_.mean(0).std(0)

                    model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]
                    reward = rewards[model_indexes, np.arange(obs.shape[0])]

                    next_obs = torch.max(torch.min(next_obs, obs_max), obs_min)
                    reward = torch.clamp(reward, rew_min, rew_max)
                    
                    print('average reward:', reward.mean().item())
                    print('average uncertainty:', uncertainty.mean().item())

                    penalized_reward = reward - self.args['lam'] * uncertainty
                    # dones = torch.zeros_like(reward)
                    terminals = self.terminal_fn(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy())
                    nonterm_mask = (~terminals).flatten()

                    batch_data = Batch({
                        "obs" : obs.cpu(),
                        "act" : action.cpu(),
                        "rew" : penalized_reward.cpu(),
                        "done" : terminals,
                        "obs_next" : next_obs.cpu(),
                    })

                    model_buffer.put(batch_data)

                    if nonterm_mask.sum() == 0:
                        break

                    obs = next_obs[nonterm_mask]

            # update
            for _ in range(self.args['steps_per_epoch']):
                batch = train_buffer.sample(real_batch_size)
                model_batch = model_buffer.sample(model_batch_size)
                batch = Batch.cat([batch, model_batch], axis=0)
                batch.to_torch(device=self.device)

                self._sac_update(batch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            res = callback_fn(self.get_policy())
            
            res['uncertainty'] = uncertainty.mean().item()
            res['reward'] = reward.mean().item()
            self.log_res(epoch, res)

        return self.get_policy()

    def _sac_update(self, batch_data):
        obs = batch_data['obs']
        action = batch_data['act']
        next_obs = batch_data['obs_next']
        reward = batch_data['rew']
        done = batch_data['done']

        # update critic
        obs_action = torch.cat([obs, action], dim=-1)
        qs = torch.stack([critic(obs_action) for critic in self.critics], 0)

        with torch.no_grad():
            if self.args['max_q_backup']:
                batch_size = obs.shape[0]
                tmp_next_obs = next_obs.unsqueeze(1) \
                    .repeat(1, 10, 1) \
                    .view(batch_size * 10, next_obs.shape[-1])
                tmp_next_action = self.actor(tmp_next_obs).sample()
                tmp_next_obs_action = torch.cat([tmp_next_obs, tmp_next_action], dim=-1)
                tmp_next_qs = torch.cat([target_critic(tmp_next_obs_action) for target_critic in self.target_critics], 1)
                tmp_next_qs = tmp_next_qs.view(batch_size, 10, len(self.target_critics)).max(1)[0].view(-1, len(self.target_critics))
                next_q = torch.min(tmp_next_qs, 1)[0].reshape(-1, 1)
            else:
                next_action_dist = self.actor(next_obs)
                next_action = next_action_dist.sample()
                log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
                next_obs_action = torch.cat([next_obs, next_action], dim=-1)
                next_qs = torch.cat([target_critic(next_obs_action) for target_critic in self.target_critics], 1)
                next_q = torch.min(next_qs, 1)[0].reshape(-1, 1)
                if not self.args['deterministic_backup']:
                    alpha = torch.exp(self.log_alpha)
                    next_q -= alpha * log_prob
            y = reward + self.args['discount'] * (1 - done) * next_q
            y = torch.clamp(y, 0, None)

        critic_loss = ((qs - y) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        for target_critic, critic in zip(self.target_critics, self.critics):
            self._sync_weight(target_critic, critic, soft_target_tau=self.args['soft_target_tau'])

        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.args['target_entropy']).detach())

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

        # update actor
        action_dist = self.actor(obs)
        new_action = action_dist.rsample()
        action_log_prob = action_dist.log_prob(new_action)
        new_obs_action = torch.cat([obs, new_action], dim=-1)
        q = torch.cat([critic(new_obs_action) for critic in self.critics], 1)
        actor_loss = - torch.min(q, 1)[0].mean() + torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_transition(self, transition, data, optim):
        data.to_torch(device=self.device)
        dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
        loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
        loss = loss.mean()

        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_transition(self, transition, valdata):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
            loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1,2))
            return list(loss.cpu().numpy())