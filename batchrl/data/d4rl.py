import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from batchrl.utils.data import Batch
from batchrl.data import dataset_dir
from batchrl.utils.io import save_pkl,load_pkl
        
        
class D4RLBatch(Batch):
    def sample(self,batch_size):
        length = len(self)
        assert 1 <= batch_size
        
        indices = np.random.randint(0, length, batch_size)
        
        return self[indices]
    
def load_d4rl_data(task):
    env = gym.make(task)
    if 'random-expert' in task:
        dataset = d4rl.basic_dataset(env)
    else:
        dataset = d4rl.qlearning_dataset(env)
        

    buffer = D4RLBatch(
        obs = dataset['observations'],
        obs_next = dataset['next_observations'],
        act = dataset['actions'],
        rew = np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done = np.expand_dims(np.squeeze(dataset['terminals']), 1),
        #done = np.expand_dims(np.array([0]), 1),
        )

    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer


def load_d4rl_data_by_step(task, step_num):
    assert step_num > 0
    env = gym.make(task)
    if 'random-expert' in task:
        dataset = d4rl.basic_dataset(env)
    else:
        dataset = d4rl.qlearning_dataset(env)
    if step_num > 0:
        step_slice = slice(0,min(step_num,len(dataset['terminals'])),)
    buffer = D4RLBatch(
        obs = dataset['observations'][step_slice],
        obs_next = dataset['next_observations'][step_slice],
        act = dataset['actions'][step_slice],
        rew = np.expand_dims(np.squeeze(dataset['rewards'][step_slice]), 1),
        done = np.expand_dims(np.squeeze(dataset['terminals'][step_slice]), 1),
        #done = np.expand_dims(np.array([0]), 1),
        )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    
    
    return buffer

def load_d4rl_buffer(task, episode_num=None, step_num=None):
    offline_buffer_file = os.path.join(dataset_dir, task + ".pkl")
    if episode_num is not None:
        offline_buffer_file = os.path.join(dataset_dir, task + "-episode" + str(episode_num) + ".pkl")
    if step_num is not None:
        offline_buffer_file = os.path.join(dataset_dir, task + "-step" + str(step_num) + ".pkl")
        
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if os.path.exists(offline_buffer_file):
        logger.info('Load offline buffer from temporary file : {}', offline_buffer_file)
        offline_buffer = load_pkl(offline_buffer_file)
    else:
        logger.info('Load d4rl dataset : {}', task)
        offline_buffer = load_d4rl_data(task,)
        if episode_num is not None:
            offline_buffer = load_d4rl_data(task, episode_num = episode_num)
        if step_num is not None:
             offline_buffer = load_d4rl_data_by_step(task,step_num=step_num)
             
        logger.info('Save offline buffer as temporary file : {}', offline_buffer_file)
        save_pkl(offline_buffer, offline_buffer_file)
    
    logger.info('Buffer Length: {}', len(offline_buffer))
    return offline_buffer