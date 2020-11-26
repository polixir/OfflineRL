import gym
import d4rl

def get_env_dim(task):
    env = gym.make(task)
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size
    
    return obs_dim, act_dim

