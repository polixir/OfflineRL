# MOPO: Model-based Offline Policy Optimization
# https://arxiv.org/abs/2005.13239
# https://github.com/tianheyu927/mopo

import torch
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.functional import F

from tianshou.data import Batch

from batchrl.algo.base import BaseAlgo
from batchrl.utils.data import to_torch, sample
from batchrl.utils.net.common import MLP, Net
from batchrl.utils.net.tanhpolicy import TanhGaussianPolicy

def algo_init(args):
    logger.info('Run algo_init function')
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from batchrl.utils.env import get_env_shape
        obs_shape, action_shape = get_env_shape(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    transition_creator = lambda: Transition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers']).to(args['device'])
    transitions = [transition_creator() for i in range(args['transition_init_num'])]
    transition_optims = [torch.optim.Adam(model.parameters(), lr=args['transition_lr']) for model in transitions]

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

    q1 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    q2 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['actor_lr'])

    return {
        "transitions" : {"net" : transitions, "opt" : transition_optims},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer},
        "critic" : {"net" : [q1, q2], "opt" : critic_optim},
    }

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class Transition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 2 * (obs_dim + 1), hidden_features, hidden_layers, norm=None, hidden_activation='swish')
        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + 1) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + 1) * -5, requires_grad=True))

    def forward(self, obs_action):
        mu, logstd = torch.chunk(self.net(obs_action), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        return torch.distributions.Normal(mu, torch.exp(logstd))


class MOPOBuffer:
    def __init__(self, buffer_size):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data):
        batch_data.to_torch(device='cpu')

        if self.data is None:
            self.data = batch_data
        else:
            self.data.cat_(batch_data)
        
        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size : ]

    def __len__(self):
        if self.data is None: return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.transitions = algo_init['transitions']['net']
        self.transition_optims = algo_init['transitions']['opt']
        self.selected_transitions = None

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.device = args['device']
        
    def train(self, buffer, callback_fn):
        self.selected_transitions = self.train_transitions(buffer)
        for transition in self.selected_transitions: transition.requires_grad_(False)   
        policy = self.train_policy(buffer, self.selected_transitions, callback_fn)

    def save_model(self, model_save_path):
        torch.save(self.get_policy(), model_save_path)
    
    def get_policy(self):
        return self.actor

    def train_transitions(self, buffer):
        data_size = len(buffer)
        val_size = int(data_size * 0.01) + 1
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]

        val_losses = [float('inf') for i in range(len(self.transitions))]
        output_transitions = [None] * len(self.transitions)

        epoch = 0
        cnt = 0
        while True:
            for _ in range(self.args['transition_steps_per_epoch']):
                batch = sample(train_buffer, self.args['batch_size'])
                for transition, optim in zip(self.transitions, self.transition_optims):
                    self._train_transition(transition, batch, optim)
            new_val_losses = [self._eval_transition(transition, valdata) for transition in self.transitions]
            print(new_val_losses)

            change = False
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                if new_loss < old_loss:
                    change = True
                    val_losses[i] = new_loss
                    output_transitions[i] = deepcopy(self.transitions[i])

            if change:
                cnt = 0
            else:
                cnt += 1

            if cnt >= 3:
                break

        return self._select_best(val_losses, output_transitions, n=self.args['transition_select_num'])

    def train_policy(self, real_buffer, transitions, callback_fn):
        real_batch_size = int(self.args['batch_size'] * self.args['real_data_ratio'])
        model_batch_size = self.args['batch_size']  - real_batch_size
        
        model_buffer = MOPOBuffer(self.args['buffer_size'])

        for epoch in range(self.args['max_epoch']):
            # collect data
            with torch.no_grad():
                obs = real_buffer.sample(int(self.args['data_collection_per_epoch']))['obs']
                obs = torch.tensor(obs, device=self.device)
                for t in range(self.args['horizon']):
                    action = self.actor(obs).sample()
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = [transition(obs_action) for transition in transitions]
                    dist_stds = torch.stack([dist.stddev for dist in next_obs_dists], dim=0)
                    uncertainty = torch.max(dist_stds, dim=0)[0].sum(dim=-1, keepdim=True)
                    next_obses = torch.stack([dist.sample() for dist in next_obs_dists], dim=0)
                    model_indexes = np.random.randint(0, len(transitions), size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]
                    rewards = next_obs[:, -1:] - self.args['lam'] * uncertainty
                    next_obs = next_obs[:, :-1] + obs
                    dones = torch.zeros_like(rewards)

                    batch_data = Batch({
                        "obs" : obs.cpu(),
                        "act" : action.cpu(),
                        "rew" : rewards.cpu(),
                        "done" : dones.cpu(),
                        "obs_next" : next_obs.cpu(),
                    })

                    model_buffer.put(batch_data)

                    obs = next_obs

            # update
            for _ in range(self.args['steps_per_epoch']):
                batch = real_buffer.sample(real_batch_size)
                model_batch = model_buffer.sample(model_batch_size)
                batch.cat_(model_batch)
                batch.to_torch(device=self.device)

                self._sac_update(batch)

            res = callback_fn(self.get_policy())
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
        _q1 = self.q1(obs_action)
        _q2 = self.q2(obs_action)

        with torch.no_grad():
            next_action_dist = self.actor(next_obs)
            next_action = next_action_dist.sample()
            log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            next_obs_action = torch.cat([next_obs, next_action], dim=-1)
            _target_q1 = self.target_q1(next_obs_action)
            _target_q2 = self.target_q2(next_obs_action)
            alpha = torch.exp(self.log_alpha)
            y = reward + self.args['discount'] * (1 - done) * (torch.min(_target_q1, _target_q2) - alpha * log_prob)

        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

        # # update alpha
        # alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.args['target_entropy']).detach())

        # self.log_alpha_optim.zero_grad()
        # alpha_loss.backward()
        # self.log_alpha_optim.step()

        # update actor
        action_dist = self.actor(obs)
        new_action = action_dist.rsample()
        action_log_prob = action_dist.log_prob(new_action)
        new_obs_action = torch.cat([obs, new_action], dim=-1)
        q = torch.min(self.q1(new_obs_action), self.q2(new_obs_action))
        actor_loss = - q.mean() + torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def _select_best(self, metrics, models, n):
        pairs = [(metric, model) for metric, model in zip(metrics, models)]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_models = [pairs[i][1] for i in range(n)]
        return selected_models

    def _train_transition(self, transition, data, optim):
        data.to_torch(device=self.device)
        dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
        loss = - dist.log_prob(torch.cat([data['obs_next'] - data['obs'], data['rew']], dim=-1))
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_transition(self, transition, valdata):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
            loss = ((dist.mean - torch.cat([valdata['obs_next'] - valdata['obs'], valdata['rew']], dim=-1)) ** 2).mean()
            return loss.mean().item()
