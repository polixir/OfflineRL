import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from batchrl.utils.data import Batch
from batchrl.data import dataset_dir
from batchrl.utils.data import SampleBatch,get_scaler

    
def load_revive_buffer(data_dir):
    data = np.load(data_dir)

    buffer = SampleBatch(
        obs = data["obs"],
        obs_next = data["next_obs"],
        act = data["action"],
        rew = data["reward"],
        done = data["done"],
        )
    
    rew_scaler = get_scaler(buffer.rew)
    buffer.rew = rew_scaler.transform(buffer.rew)
    #buffer.rew =  buffer.rew * 0.01
    #buffer.done[buffer.rew < np.sort(buffer.rew.reshape(-1))[int(len(buffer)*0.01)]] = 1
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    
    
    return buffer
