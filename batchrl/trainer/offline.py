import torch
import gtimer as gt
import numpy as np

class OfflineTrainer():
    def __init__(
            self,
            algo_runner,
            history_buffer,
            args,
    ):
        self.algo_runner = algo_runner
        self.replay_buffer = history_buffer
        
        self._start_epoch = 0
        
    def train(self,):        
        for epoch in range(1,args["max_epoch"]+1):
            for step in range(1,args["steps_per_epoch"]+1):
                train_data = self.replay_buffer.sample(args["batch_size"])
                self.algo_runner.train(train_data)
            self.eval(epoch)
            
    def algo_train(self):
        self.algo_runner.train(self.replay_buffer)
        
    def eval(self, epoch, n_episode=10):
        import gym
        from tianshou.data import Collector
        from tianshou.trainer import test_episode
        from tianshou.env import SubprocVectorEnv
        from batchrl.utils.net.tanhpolicy import MakeDeterministic
        from batchrl.evaluation.d4rl import d4rl_score
        
        eval_policy = MakeDeterministic(self.algo_runner.actor,args["device"])
        eval_envs = SubprocVectorEnv([lambda: gym.make(args["task"]) for _ in range(10)])
        eval_collector = Collector(eval_policy, eval_envs)

        result = test_episode(eval_policy, eval_collector, None, epoch=epoch,n_episode=n_episode)
        
        score = d4rl_score(self.task, result["rew"], result["len"])