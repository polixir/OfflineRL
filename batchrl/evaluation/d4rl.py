import gym
import d4rl 
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE
from tianshou.data import to_numpy,to_torch

def d4rl_score(task, rew_mean, len_mean):
    score = (rew_mean - REF_MIN_SCORE[task]) / (REF_MAX_SCORE[task] - REF_MIN_SCORE[task]) * 100
    
    return score


def d4rl_eval_fn(task, eval_episodes=100):
    env = gym.make(task)
    
    def d4rl_eval(policy):
        episode_rewards = []
        episode_lengths = []
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]
                action = policy.get_action(to_torch(state[np.newaxis], device=next(policy.parameters()).device, dtype=torch.float32))
                action= action[0]
                state, reward, done, _ = env.step(action)
                rewards += reward
                lengths += 1

            episode_rewards.append(rewards)
            episode_lengths.append(lengths)


        rew_mean = np.mean(episode_rewards)
        len_mean = np.mean(episode_lengths)
        
        score = d4rl_score(task, rew_mean, len_mean)
        
        res = OrderedDict()
        res["Reward Mean"] = rew_mean
        res["Mean episode length"] = len_mean
        res["D4rl Score"] = score

        return res
    
    return d4rl_eval