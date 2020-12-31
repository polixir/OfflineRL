import gym


"""
def get_env(task):
    if task.startswith("ib"):
        from industrial_benchmark_python.IBGym import IBGym
        env = IBGym(setpoint=70, reward_type='classic', action_type='continuous', observation_type='include_past') 
    elif task.startswith("traffic"):
        from offlinedata.get_env import create_env
        env = create_env("traffic")
    else:
        if task.startswith("HalfCheetah-v3"):
            env = gym.make("HalfCheetah-v3", exclude_current_positions_from_observation=False)
        elif task.startswith("Hopper-v3"):
            env = gym.make('Hopper-v3', exclude_current_positions_from_observation=False)
        elif task.startswith("Walker2d-v3"):   
            env = gym.make('Walker2d-v3',  exclude_current_positions_from_observation=False)
        else:
            try:
                import d4rl
                env = gym.make(task)
            except:
                raise NotImplementedError
        
    return env
"""
def get_env(task):
    from offlinedata.get_env import create_env
    if task.startswith("ib"):
        env = create_env("ib")
    elif task.startswith("traffic"):
        env = create_env("traffic")
    elif task.startswith("finance"):
        env = create_env("finance")
    else:
        if task.startswith("HalfCheetah-v3"):
            #env = gym.make("HalfCheetah-v3", exclude_current_positions_from_observation=False)
            env = create_env("HalfCheetah-v3")
        elif task.startswith("Hopper-v3"):
            #env = gym.make('Hopper-v3', exclude_current_positions_from_observation=False)
            env = create_env("Hopper-v3")
        elif task.startswith("Walker2d-v3"):   
            #env = gym.make('Walker2d-v3',  exclude_current_positions_from_observation=False)
            env = create_env("Walker2d-v3")
        else:
            try:
                import d4rl
                env = gym.make(task)
            except:
                raise NotImplementedError
        
    return env


def get_env_shape(task):
    env = get_env(task)
    obs_dim = env.observation_space.shape
    action_space = env.action_space
    
    if len(obs_dim) == 1:
        obs_dim = obs_dim[0]
        
    if hasattr(env.action_space, 'n'):
        act_dim = env.action_space.n
    else:
        act_dim = action_space.shape[0]
    
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




