import gym
import d4rl

def get_env_shape(task):
    env = gym.make(task)
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size
    
    return obs_dim, act_dim


def get_env_action_range(task):
    env = gym.make(task)
    act_max = float(env.action_space.high[0])
    act_min = float(env.action_space.low[0])
    
    return act_max, act_min
    
    
def get_env_state_range(task):
    env = gym.make(task)
    obs_max = float(env.observation_space.high[0])
    obs_min = float(env.observation_space.low[0])
    
    return obs_max, obs_min

