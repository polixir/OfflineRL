import os
import time
import random
from loguru import logger

from offlinerl.utils.logger import log_path
from offlinerl.utils.io import create_dir, download_helper, read_json

from offlinerl.data.newrl import load_newrl_buffer

dataset_dir = os.path.join(log_path(),"./offlinerl_datasets")
create_dir(dataset_dir)

def load_data_from_newrl(task, task_data_type = "low", task_train_num = 99):
    import newrl
    env = newrl.make(task)
    train_data, val_data = env.get_dataset(data_type = task_data_type, train_num = task_train_num)
    
    train_buffer, val_buffer = load_newrl_buffer(train_data), load_newrl_buffer(val_data)


    return train_buffer, val_buffer