import torch
import gtimer as gt
import numpy as np
#from tianshou.data import Collector
#from tianshou.trainer import test_episode

#from d4rl_evaluate import d4rl_score
#from sac_policies import MakeDeterministic


class OfflineTrainer():
    def __init__(
            self,
            algo_runner,
            history_buffer,
            args,
    ):
        self.algo_runner = algo_runner
        self.replay_buffer = history_buffer
        self.max_epoch = args["max_epoch"]
        self.steps_per_epoch = args["steps_per_epoch"]
        self.batch_size = args["batch_size"]
        self.device = args["device"]
        self.task = args["task"]
        
        self._start_epoch = 0

    def train(self,):
        for epoch in range(1,self.max_epoch+1):
            for step in range(1,self.steps_per_epoch+1):
                train_data = self.replay_buffer.sample(self.batch_size)
                self.algo_runner.train(train_data)
            self.eval(epoch)
            
    
    def eval(self, epoch, n_episode=10):
        import gym
        from tianshou.data import Collector
        from tianshou.trainer import test_episode
        from tianshou.env import SubprocVectorEnv
        from batchrl.utils.net.tanhpolicy import MakeDeterministic
        from batchrl.evaluation.d4rl import d4rl_score
        
        eval_policy = MakeDeterministic(self.algo_runner.actor,self.device)
        eval_envs = SubprocVectorEnv([lambda: gym.make(self.task) for _ in range(10)])
        eval_collector = Collector(eval_policy, eval_envs)

        result = test_episode(eval_policy, eval_collector, None, epoch=epoch,n_episode=n_episode)
        
        score = d4rl_score(self.task, result["rew"], result["len"])