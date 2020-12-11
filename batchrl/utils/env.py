import gym

def get_env(task):
    if task == "ib":
        from industrial_benchmark_python.IBGym import IBGym
        env = IBGym(setpoint=70, reward_type='classic', action_type='continuous', observation_type='include_past')    
    else:
        import d4rl
        env = gym.make(task)

    return env

def get_env_shape(task):
    env = get_env(task)
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size
    
    return obs_dim, act_dim


def get_env_action_range(task):
    env = get_env(task)
    act_max = float(env.action_space.high[0])
    act_min = float(env.action_space.low[0])
    
    return act_max, act_min
    
    
def get_env_state_range(task):
    env = get_env(task)
    obs_max = float(env.observation_space.high[0])
    obs_min = float(env.observation_space.low[0])
    
    return obs_max, obs_min




