import ray
import torch

from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from offlinerl.utils.env import get_env
from offlinerl.utils.net.common import MLP
from offlinerl.evaluation.neorl import test_on_real_env
from offlinerl.evaluation.fqe import FQE

class CallBackFunction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_initialized = False
    
    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        self.is_initialized = True

    def __call__(self, policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before calls."
        raise NotImplementedError

class PeriodicCallBack(CallBackFunction):
    '''This is a wrapper for callbacks that are only needed to perform periodically.'''
    def __init__(self, callback : CallBackFunction, period : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback
        self.period = period
        self.call_count = 0

    def __getattr__(self, name : str):
        return getattr(self._callback, name)

    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        self._callback.initialize(train_buffer, val_buffer, *args, **kwargs)

    def __call__(self, policy) -> dict:
        assert self._callback.is_initialized, "`initialize` should be called before calls."
        self.call_count += 1
        if self.call_count % self.period == 0:
            return self._callback(policy)
        else:
            return {}

class CallBackFunctionList(CallBackFunction):
    # TODO: run `initialize` and `__call__` in parallel
    def __init__(self, callback_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_list = callback_list

    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        for callback in self.callback_list:
            callback.initialize(train_buffer, val_buffer, *args, **kwargs)
        self.is_initialized = True

    def __call__(self, policy) -> dict:
        eval_res = OrderedDict()

        for callback in self.callback_list:
            eval_res.update(callback(policy))

        return eval_res

class OnlineCallBackFunction(CallBackFunction):
    def initialize(self, train_buffer, val_buffer, task, number_of_runs=100, *args, **kwargs):
        self.task = task
        self.env = get_env(self.task)
        self.is_initialized = True
        self.number_of_runs = number_of_runs

    def __call__(self, policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before callback."
        policy = deepcopy(policy).cpu()
        eval_res = OrderedDict()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        eval_res.update(test_on_real_env(policy, self.env, number_of_runs=self.number_of_runs))
        return eval_res

class FQECallBackFunction(CallBackFunction):
    def initialize(self, train_buffer=None, val_buffer=None, start_index=None, pretrain=False, *args, **kwargs):
        assert train_buffer is not None or val_buffer is not None, 'you need to provide at least one buffer to run FQE test'
        self.buffer = val_buffer or train_buffer
        self.start_index = start_index
        self.pretrain = pretrain

        if self.pretrain:
            '''implement a base value function here'''
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # clone the behaviorial policy
            data = self.buffer[0]
            policy = MLP(data.obs.shape[-1], data.act.shape[-1], 1024, 2).to(device)
            optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
            for i in tqdm(range(10000)):
                data = self.buffer.sample(256)
                data = data.to_torch(device=device)
                _act = policy(data.obs)
                loss = ((data.act - _act) ** 2).mean()
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            policy.get_action = lambda x: policy(x)
            fqe = FQE(policy, self.buffer, device=device)
            self.init_critic = fqe.train_estimator(num_steps=100000)
        else:
            self.init_critic = None

        self.is_initialized = True

    def __call__(self, policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before callback."
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        policy = deepcopy(policy)
        policy = policy.to(device)

        Fqe = FQE(policy, self.buffer,
                  q_hidden_features=1024,
                  q_hidden_layers=4,
                  device=device)

        if self.pretrain:
            critic = Fqe.train_estimator(self.init_critic, num_steps=100000)
        else:
            critic = Fqe.train_estimator(num_steps=250000)

        if self.start_index is not None:
            data = self.buffer[self.start_index]
            obs = data.obs
            obs = torch.tensor(obs).float()
            batches = torch.split(obs, 256, dim=0)
        else:
            batches = [torch.tensor(self.buffer.sample(256).obs).float() for _ in range(100)]

        estimate_q0 = []
        with torch.no_grad():
            for o in batches:
                o = o.to(device)
                a = policy.get_action(o)
                init_sa = torch.cat((o, a), -1).to(device)
                estimate_q0.append(critic(init_sa).cpu())
        estimate_q0 = torch.cat(estimate_q0, dim=0)

        res = OrderedDict()
        res["FQE"] = estimate_q0.mean().item()
        return res

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
            data = data.to_torch(device=device)
            next = self.trainsition(torch.cat([data.obs, data.act], dim=-1))
            loss = ((next - torch.cat([data.rew, data.obs_next], dim=-1)) ** 2).mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        self.is_initialized = True

    def __call__(self, policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before callback."
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        eval_size = 10000
        batch = self.buffer[:eval_size]
        data = batch.to_torch(dtype=torch.float32, device=device)

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

class AutoOPECallBackFunction(CallBackFunction):
    def initialize(self, train_buffer=None, val_buffer=None, *args, **kwargs):
        assert train_buffer is not None or val_buffer is not None, 'you need to provide at least one buffer to run AutoOPE test'
        self.buffer = val_buffer or train_buffer

        '''implement a model here'''

        from offlinerl.algo.modelbase.mopo import soft_clamp
        import numpy as np
        from torch.functional import F

        logstd_MIN = -20
        logstd_MAX = 2

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class AutoRegressiveTransition(torch.nn.Module):
            def __init__(self, obs_dim, action_dim, device,
                         with_reward=True,
                         norm: str = 'bn',
                         hidden_features=1024,
                         hidden_layers=4
                         ):

                super(AutoRegressiveTransition, self).__init__()

                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.device = device
                # 是否需要生成 r
                self.with_reward = with_reward

                mask = 1 - np.triu(np.ones([obs_dim, obs_dim]), 1) - np.eye(obs_dim)
                if with_reward:
                    mask = np.concatenate([mask, np.ones([1, obs_dim])], axis=0)

                """ 
                mask : 
                
                [0 0 0 0
                 1 0 0 0
                 1 1 0 0
                 1 1 1 0
                 1 1 1 1]
                """

                self.mask = torch.from_numpy(mask).float().to(self.device)
                self.one_hot = torch.eye(obs_dim, device=self.device)
                """ 
                one_hot  : 

                [1 0 0
                 0 1 0
                 0 0 1
                 1 1 1]
                """

                if with_reward:
                    self.one_hot = torch.cat([self.one_hot, torch.ones((1, obs_dim), device=device)], dim=0)

                self._net = MLP(
                    in_features=3 * obs_dim + action_dim,
                    out_features=2,
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    norm=norm,
                    hidden_activation='relu',
                    output_activation='identity')
                print('Auto:\n', self)

            def forward(self, s, a):
                """
                return : s_ or delta_s
                """

                s, a = self.input_check(s, a)
                sa = torch.cat([s, a], dim=-1)

                s_res = torch.zeros([sa.shape[0], self.obs_dim], device=self.device)
                for i in range(self.obs_dim):
                    one_hot = F.one_hot(torch.tensor(i, device=self.device), num_classes=self.obs_dim)
                    one_hot = torch.unsqueeze(one_hot, dim=0).expand(sa.shape[0], -1)
                    sas = torch.cat([one_hot, sa, s_res], dim=-1)
                    dist_i = self._to_dist(self._net(sas))
                    s_i = dist_i.sample()
                    s_res[:, i] = s_i.squeeze()

                return s_res

            def forward_r(self, s, a):
                """
                return : ( s_ or delta_s, reward )
                """
                s, a = self.input_check(s, a)
                sa = torch.cat([s, a], dim=-1)
                s_res = torch.zeros([sa.shape[0], self.obs_dim], device=self.device)

                for i in range(self.obs_dim):
                    one_hot = F.one_hot(torch.tensor(i, device=self.device), num_classes=self.obs_dim)
                    one_hot = torch.unsqueeze(one_hot, dim=0).expand(sa.shape[0], -1)

                    sas = torch.cat([one_hot, sa, s_res], dim=-1)
                    dist_i = self._to_dist(self._net(sas))
                    s_i = dist_i.sample()
                    s_res[:, i] = s_i.squeeze()

                final_all_one = torch.ones((sa.shape[0], self.obs_dim), device=self.device)
                sas = torch.cat([final_all_one, sa, s_res], dim=-1)
                r_dist = self._to_dist(self._net(sas))

                r = r_dist.sample()

                return s_res, r

            def log_prob_without_r(self, s, a, s_):
                sa = torch.cat([s, a], dim=-1)
                maked_s_ = torch.unsqueeze(s_, dim=1).expand(-1, self.obs_dim, -1) * self.mask

                assert maked_s_.dtype == torch.float32

                # 沿着 obs_宽度维 复制
                sa = torch.unsqueeze(sa, dim=1).expand(-1, self.obs_dim, -1)

                # 沿着batch维复制
                one_hot = torch.unsqueeze(self.one_hot, dim=0).expand(sa.shape[0], -1, -1)

                sas = torch.cat([one_hot, sa, maked_s_], dim=-1)
                dist = self._to_dist(self._net(sas))

                log_p = dist.log_prob(s_)

                return torch.sum(log_p, dim=-1)

            def log_prob_r(self, s, a, s_, r):

                sa = torch.cat([s, a], dim=-1)

                maked_s_ = torch.unsqueeze(s_, dim=1).expand(-1, self.obs_dim + 1, -1) * self.mask

                assert maked_s_.dtype == torch.float32

                # 沿着 obs_宽度维 复制
                sa = torch.unsqueeze(sa, dim=1).expand(-1, self.obs_dim + 1, -1)

                # 沿着batch维复制
                one_hot = torch.unsqueeze(self.one_hot, dim=0).expand(sa.shape[0], -1, -1)

                sas = torch.cat([one_hot, sa, maked_s_], dim=-1)
                dist = self._to_dist(self._net(sas))

                log_p = dist.log_prob(torch.cat([s_, r], dim=-1))

                return torch.sum(log_p, dim=-1)

            @staticmethod
            def _to_dist(_output):
                mu, logstd = torch.chunk(_output, 2, dim=-1)

                logstd = soft_clamp(logstd, _min=logstd_MIN, _max=logstd_MAX)

                mean = mu.squeeze()
                std = torch.exp(logstd).squeeze()

                return torch.distributions.Normal(loc=mean, scale=std)

            def save_model(self, model_save_path):
                torch.save(self, model_save_path)

            def input_check(self, *input_data):
                out = []
                for each_data in input_data:
                    if isinstance(each_data, np.ndarray):
                        each_data = torch.from_numpy(each_data.astype(np.float32)).to(self.device)
                    if each_data.dim() == 1:
                        each_data = torch.unsqueeze(each_data, dim=0)
                    if each_data.dim() > 2:
                        each_data = torch.unsqueeze(each_data.squeeze(), dim=0)
                    if each_data.device != self.device:
                        each_data = each_data.to(self.device)
                    out.append(each_data)
                return tuple(out)

        data_train = self.buffer[0]
        OUTPUT_DIM = data_train.obs.shape[-1] + 1
        self.trainsition = AutoRegressiveTransition(
            obs_dim=data_train.obs.shape[-1],
            action_dim=data_train.act.shape[-1],
            device=device,
            with_reward=True,
            norm='bn',
            hidden_features=1024,
            hidden_layers=4).to(device)

        optim_Auto = torch.optim.Adam(self.trainsition.parameters(), lr=1e-3, weight_decay=float(1e-6))
        scheduler_Auto = torch.optim.lr_scheduler.StepLR(optim_Auto, step_size=5, gamma=0.99)

        self.trainsition.train()
        for i in tqdm(range(100_0000)):
            data_train = train_buffer.sample(128)
            data_train = data_train.to_torch(device=device)

            nll_Auto = -torch.mean(
                self.trainsition.log_prob_r(data_train.obs,
                                            data_train.act,
                                            data_train.obs_next,
                                            data_train.rew)
            )

            optim_Auto.zero_grad()
            nll_Auto.backward()
            optim_Auto.step()

            if i % 1000 == 0:
                data_val = val_buffer.sample(1024)
                data_val = data_val.to_torch(device=device)
                self.trainsition.eval()
                nll_val_Auto = -torch.mean(
                    self.trainsition.log_prob_r(data_val.obs,
                                                data_val.act,
                                                data_val.obs_next,
                                                data_val.rew)
                )
                print('batch_num : {} | lr : {:.2e} |\tAuto : nll_val {:.2f} |'.format(
                        i, scheduler_Auto.get_lr()[0], nll_val_Auto / OUTPUT_DIM))
                self.trainsition.train()
                scheduler_Auto.step()

        self.is_initialized = True

    def __call__(self, policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before callback."
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        eval_size = 10000
        batch = self.buffer[:eval_size]
        batch = batch.to_torch(dtype=torch.float32, device=device)
        obs = batch.obs

        self.trainsition.eval()
        with torch.no_grad():
            ret = 0
            for t in range(200):
                action = policy.get_action(obs)
                obs, r = self.trainsition.forward_r(obs, action)
                ret += (0.995 ** t) * r

        res = OrderedDict()
        res["Auto-OPE"] = (torch.nansum(ret) / (1 - torch.isnan(ret).float()).sum()).cpu().item()
        return res


def get_defalut_callback(*args, **kwargs):
    return CallBackFunctionList([
        OnlineCallBackFunction(*args, **kwargs),
        PeriodicCallBack(FQECallBackFunction(*args, **kwargs), period=25),
        PeriodicCallBack(MBOPECallBackFunction(*args, **kwargs), period=25),
        PeriodicCallBack(AutoOPECallBackFunction(*args, **kwargs), period=25),
    ])