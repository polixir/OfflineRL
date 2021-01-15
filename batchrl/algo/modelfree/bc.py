import torch
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.functional import F
from torch.distributions import Normal, kl_divergence

from tianshou.data import Batch

from batchrl.algo.base import BaseAlgo
from batchrl.utils.data import to_torch, sample
from batchrl.utils.net.common import Net
from batchrl.utils.net.tanhpolicy import TanhGaussianPolicy

def algo_init(args):
    logger.info('Run algo_init function')
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
        max_action = args["max_action"]
    elif "task" in args.keys():
        from batchrl.utils.env import get_env_shape, get_env_action_range
        obs_shape, action_shape = get_env_shape(args['task'])
        max_action, _ = get_env_action_range(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    net_a = Net(layer_num=args['actor_layers'], 
                state_shape=obs_shape, 
                hidden_layer_size=args['actor_features'])

    actor = TanhGaussianPolicy(preprocess_net=net_a,
                               action_shape=action_shape,
                               hidden_layer_size=args['actor_features'],
                               conditioned_sigma=True).to(args['device'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    return {
        "actor" : {"net" : actor, "opt" : actor_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.batch_size = self.args['batch_size']
        self.device = self.args['device']
        
    def train(self, train_buffer, val_buffer, callback_fn):
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                batch_data.to_torch(device=self.device)
                obs = batch_data['obs']
                action = batch_data['act']

                action_dist = self.actor(obs)
                loss = - action_dist.log_prob(action).mean()

                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
            
            res = callback_fn(policy=self.get_policy(), 
                              train_buffer=train_buffer,
                              val_buffer=val_buffer,
                              args=self.args)
            
            self.log_res(epoch, res)

        return self.get_policy()
    
    def get_policy(self):
        return self.actor