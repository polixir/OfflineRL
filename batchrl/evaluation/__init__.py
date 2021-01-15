import ray
import torch

from tqdm import tqdm
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import OrderedDict
from tianshou.data import to_numpy, to_torch

from batchrl.utils.env import get_env
from batchrl.utils.net.common import MLP
from batchrl.evaluation.porl import test_on_real_env
from batchrl.evaluation.fqe import FQE, fqe_eval_fn

class CallBackFunction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.call_count = 0
        self.is_initialized = False
    
    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        self.is_initialized = True

    def __call__(self, policy):
        assert self.is_initialized, "`initialize` should be called before calls."
        self.call_count += 1

class CallBackFunctionList(CallBackFunction):
    # TODO: run `initialize` and `__call__` in parallel
    def __init__(self, callback_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_list = callback_list

    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        for callback in self.callback_list:
            callback.initialize(train_buffer, val_buffer, *args, **kwargs)
        self.is_initialized = True

    def __call__(self, policy):
        eval_res = OrderedDict()

        for callback in self.callback_list:
            eval_res.update(callback(policy))

        return eval_res

class OnlineCallBackFunction(CallBackFunction):
    def initialize(self, train_buffer, val_buffer, task, *args, **kwargs):
        self.task = task
        self.env = get_env(self.task)
        self.is_initialized = True

    def __call__(self, policy):
        assert self.is_initialized, "`initialize` should be called before callback."
        policy = deepcopy(policy).cpu()
        eval_res = OrderedDict()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        eval_res.update(test_on_real_env(policy, self.env))
        return eval_res

class FQECallBackFunction(CallBackFunction):
    def initialize(self, train_buffer=None, val_buffer=None, *args, **kwargs):
        assert train_buffer is not None or val_buffer is not None, 'you need to provide at least one buffer to run FQE test'
        self.buffer = val_buffer or train_buffer

        '''implement a base value function here'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # clone the behaviorial policy
        data = self.buffer[0]
        policy = MLP(data.obs.shape[-1], data.act.shape[-1], 1024, 2).to(device)
        optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        for i in tqdm(range(10000)):
            data = self.buffer.sample(256)
            data = to_torch(data, device=device)
            _act = policy(data.obs)
            loss = ((data.act - _act) ** 2).mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        policy.get_action = lambda x: policy(x)
        fqe = FQE(policy, self.buffer, device=device)
        self.init_critic = fqe.train_estimator(num_steps=100000)

        self.is_initialized = True

    def __call__(self, policy):
        assert self.is_initialized, "`initialize` should be called before callback."
        self.call_count += 1
        if self.call_count % 25 == 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            policy = deepcopy(policy)
            policy = policy.to(device)

            Fqe = FQE(policy, self.buffer,
                    q_hidden_features=1024,
                    q_hidden_layers=4,
                    device=device)

            critic = Fqe.train_estimator(self.init_critic, num_steps=100000)

            eval_size = 10000
            batch = self.buffer[:eval_size]
            data = to_torch(batch, torch.float32, device=device)
            o0 = data.obs
            a0 = policy.get_action(o0)
            init_sa = torch.cat((o0, a0), -1).to(device)
            with torch.no_grad():
                estimate_q0 = critic(init_sa)
            res = OrderedDict()
            res["FQE"] = estimate_q0.mean().item()
            return res
        else:
            return {}

class MBOPECallBackFunction(CallBackFunction):
    def initialize(self, train_buffer=None, val_buffer=None, *args, **kwargs):
        assert train_buffer is not None or val_buffer is not None, 'you need to provide at least one buffer to run MBOPE test'
        self.buffer = val_buffer or train_buffer

        '''implement a model here'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # learn a simple world model
        data = self.buffer[0]
        self.trainsition = MLP(data.obs.shape[-1] + data.act.shape[-1], data.obs.shape[-1] + 1, 512, 3).to(device)
        optim = torch.optim.Adam(self.trainsition.parameters(), lr=1e-3)
        for i in tqdm(range(100000)):
            data = self.buffer.sample(256)
            data = to_torch(data, device=device)
            next = self.trainsition(torch.cat([data.obs, data.act], dim=-1))
            loss = ((next - torch.cat([data.rew, data.obs_next], dim=-1)) ** 2).mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        self.is_initialized = True

    def __call__(self, policy):
        assert self.is_initialized, "`initialize` should be called before callback."
        self.call_count += 1
        if self.call_count % 25 == 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            eval_size = 10000
            batch = self.buffer[:eval_size]
            data = to_torch(batch, torch.float32, device=device)

            with torch.no_grad():
                gamma = 0.99
                reward = 0
                obs = data.obs
                for t in range(20):
                    action = policy.get_action(obs)
                    next = self.trainsition(torch.cat([obs, action], dim=-1))
                    r = next[..., 0]
                    obs = next[..., 1:]
                    reward += gamma ** t * r

            res = OrderedDict()
            res["MB-OPE"] = reward.mean().item()
            return res
        else:
            return {}

def get_defalut_callback(*args, **kwargs):
    return CallBackFunctionList([
        OnlineCallBackFunction(*args, **kwargs),
        FQECallBackFunction(*args, **kwargs),
        MBOPECallBackFunction(*args, **kwargs),
    ])