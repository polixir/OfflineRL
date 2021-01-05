import multiprocessing
import gym
import ray
from copy import deepcopy
import numpy as np
from collections import OrderedDict

from batchrl.utils.env import get_env


@ray.remote
def test_one_trail(env, policy):
    env = deepcopy(env)
    policy = deepcopy(policy)

    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state).reshape(-1)
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1

    return (rewards, lengths)

def test_on_real_env(policy, env, number_of_runs=10):
    rewards = []
    episode_lengths = []
    #policy = deepcopy(policy).cpu()

    results = ray.get([test_one_trail.remote(env, policy) for _ in range(number_of_runs)])
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]

    rew_mean = np.mean(rewards)
    len_mean = np.mean(episode_lengths)

    res = OrderedDict()
    res["Reward_Mean_Env"] = rew_mean
    res["Length_Mean_Env"] = len_mean

    return res


"""
def gym_policy_eval(task, eval_episodes=100):
    env = get_env(task)
    ray.init(ignore_reinit_error=True)
    def test_on_real_env(policy, number_of_runs=10):
        rewards = []
        episode_lengths = []
        policy = deepcopy(policy).cpu()
        
        results = ray.get([test_one_trail.remote(env, policy) for _ in range(number_of_runs)])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]

        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean"] = rew_mean
        res["Length_Mean"] = len_mean

        return res

    return test_on_real_env

"""
