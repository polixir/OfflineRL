import os
import pickle

import gym
import d4rl
import numpy as np

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
        )
    print ('Number of terminals on: ', buffer.done.sum())

    return buffer

def load_d4rl_buffer(task):
    offline_buffer_file = os.path.join(dataset_dir, task + ".pkl")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if os.path.exists(offline_buffer_file):
        print("Load offline buffer -> ", offline_buffer_file)
        offline_buffer = load_pkl(offline_buffer_file)
    else:
        print("Load d4rl dataset -> ", task)
        offline_buffer = load_d4rl_data(task)
        print("Save offline buffer -> ", offline_buffer_file)
        save_pkl(offline_buffer, offline_buffer_file)
        
    return offline_buffer