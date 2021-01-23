import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from batchrl.utils.data import SampleBatch, get_scaler
# from batchrl.data import dataset_dir
# from batchrl.utils.io import save_pkl, load_pkl
    
def load_d4rl_buffer(task):
    env = gym.make(task)
    dataset = d4rl.qlearning_dataset(env)

    buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )

    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer

# def load_d4rl_by_episode(task, episode_num):
#     env = gym.make(task)
#     if 'random-expert' in task:
#         dataset = d4rl.basic_dataset(env)
#     else:
#         dataset = d4rl.qlearning_dataset(env, terminate_on_end=True)
        
    
#     done_index = np.argwhere(dataset['terminals'] == True)
#     stop_index = len(dataset['terminals']) + 1
    
#     if episode_num < len(done_index):
#         stop_index = done_index[episode_num-1][0] + 1 
#     episode_slice = slice(0,stop_index)
    
    
#     buffer = D4RLBatch(
#         obs = dataset['observations'][episode_slice],
#         obs_next = dataset['next_observations'][episode_slice],
#         act = dataset['actions'][episode_slice],
#         rew = np.expand_dims(np.squeeze(dataset['rewards'][episode_slice]), 1),
#         done = np.expand_dims(np.squeeze(dataset['terminals'][episode_slice]), 1),
#         #done = np.expand_dims(np.array([0]), 1),
#         )
#     if episode_num > 0:
#         assert buffer.done.sum() <= episode_num
        
#     logger.info('Number of terminals on: {}', np.sum(buffer.done))
#     return buffer


# def load_d4rl_data_by_step(task, step_num):
#     assert step_num > 0
#     env = gym.make(task)
#     if 'random-expert' in task:
#         dataset = d4rl.basic_dataset(env)
#     else:
#         dataset = d4rl.qlearning_dataset(env)
#     if step_num > 0:
#         step_slice = slice(0,min(step_num,len(dataset['terminals'])),)
#     buffer = D4RLBatch(
#         obs = dataset['observations'][step_slice],
#         obs_next = dataset['next_observations'][step_slice],
#         act = dataset['actions'][step_slice],
#         rew = np.expand_dims(np.squeeze(dataset['rewards'][step_slice]), 1),
#         done = np.expand_dims(np.squeeze(dataset['terminals'][step_slice]), 1),
#         #done = np.expand_dims(np.array([0]), 1),
#         )
#     logger.info('Number of terminals on: {}', np.sum(buffer.done))
    
    
#     return buffer

# def load_d4rl_buffer(task, episode_num=None, step_num=None):
#     offline_buffer_file = os.path.join(dataset_dir, task + ".pkl")
#     if episode_num is not None:
#         offline_buffer_file = os.path.join(dataset_dir, task + "-episode" + str(episode_num) + ".pkl")
#     if step_num is not None:
#         offline_buffer_file = os.path.join(dataset_dir, task + "-step" + str(step_num) + ".pkl")
        
#     if not os.path.exists(dataset_dir):
#         os.makedirs(dataset_dir)
#     if os.path.exists(offline_buffer_file):
#         logger.info('Load offline buffer from temporary file : {}', offline_buffer_file)
#         offline_buffer = load_pkl(offline_buffer_file)
#     else:
#         logger.info('Load d4rl dataset : {}', task)
#         offline_buffer = load_d4rl_data(task,)
#         if episode_num is not None:
#             offline_buffer = load_d4rl_by_episode(task, episode_num = episode_num)
#         if step_num is not None:
#              offline_buffer = load_d4rl_data_by_step(task,step_num=step_num)
             
#         logger.info('Save offline buffer as temporary file : {}', offline_buffer_file)
#         save_pkl(offline_buffer, offline_buffer_file)
    
#     logger.info('Buffer Length: {}', len(offline_buffer))
#     return offline_buffer