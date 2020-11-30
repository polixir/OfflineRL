import gym
import d4rl 
import numpy as np
from tqdm import tqdm
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE
from tianshou.data import to_numpy

def d4rl_score(task, rew_mean, len_mean):
    # Output result
    print('D4rl test Result:')
    print('-' * 30)
    print(f'Reward: {rew_mean}')
    Score = (rew_mean - REF_MIN_SCORE[task]) / (REF_MAX_SCORE[task] - REF_MIN_SCORE[task]) * 100
    print(f'Score: {Score}')
    print(f'Average Length: {len_mean}')
    print('-' * 30)
    return Score


def d4rl_eval(task, policy, number_of_runs=10, ):
    env = gym.make(task)
    rewards = []
    episode_lengths = []
    
    for i in range(number_of_runs):
        obs = env.reset()
        reward = 0
        length = 0
        while True:
            action = policy.get_action(obs[np.newaxis])[0]
            state, r, done, info = env.step(action)
            reward += r
            length += 1

            if done:
                break

        episode_lengths.append(length)
        rewards.append(reward)

    rew_mean = np.mean(rewards)
    len_mean = np.mean(episode_lengths)
    
    d4rl_score(task, rew_mean, len_mean)