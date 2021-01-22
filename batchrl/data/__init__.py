import os
import time
import random
from loguru import logger

from batchrl.utils.logger import log_path
from batchrl.utils.io import create_dir, download_helper, read_json

from batchrl.data.revive import load_revive_buffer

dataset_dir = os.path.join(log_path(),"./batchrl_datasets")
create_dir(dataset_dir)

def get_eval_data_name(train_data_name):
    eval_data_name = train_data_name.replace("train", "val")
    if "9999" in train_data_name:
        eval_data_name = eval_data_name.replace("9999", "1000")
    elif "999" in train_data_name:
        eval_data_name = eval_data_name.replace("999", "100")
    elif "99" in train_data_name:
        eval_data_name = eval_data_name.replace("99", "10")
    else:
        pass
    
    return eval_data_name

def load_data_by_task(task):
    if task.startswith("d4rl"):
        from batchrl.data.d4rl import load_d4rl_buffer
        
        task = task[5:]
        train_buffer = load_d4rl_buffer(task)
        val_buffer = None
    else:
        data_map_path = os.path.join(dataset_dir, "data_map.json")
        
        if not os.path.exists(data_map_path):
            url = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"
            data = download_helper(url, data_map_path)
            
        data_map = read_json(data_map_path)
        if task not in data_map.keys():
            url = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"
            data = download_helper(url, data_map_path)
            data_map = read_json(data_map_path)
            
        task_name = [i for i in data_map.keys() if i.startswith(task) and i.endswith(".npz")]
        if len(task_name) == 0:
            task_name = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir) if i.startswith(task) and i.endswith(".npz")]
         
        if len(task_name) == 0: 
            #logger.info('No task dataset: {}, Pleace check the task name!', task)
            raise RuntimeError('No task dataset: {}, Pleace check the task name!', task)
        elif len(task_name) > 1:
            #logger.info('Please check your task name!', task)
            raise RuntimeError('Please check your task name : {}, there are multi dataset use the name!', task)
        else:
            train_data_name = task_name[0]
            train_data_path = os.path.join(dataset_dir, train_data_name)
            eval_data_name = get_eval_data_name(task_name[0])
            eval_data_path = os.path.join(dataset_dir, eval_data_name)
               
            if not os.path.exists(train_data_path):
                data_url = data_map[train_data_name]
                logger.info('Download {}', data_url)
                download_res = download_helper(data_url, train_data_path)
            train_buffer = load_revive_buffer(train_data_path)
                
            try:
                if not os.path.exists(eval_data_path):
                    data_url = data_map[eval_data_name]
                    logger.info('Download {}', data_url)
                    download_res = download_helper(data_url, eval_data_path)

                val_buffer = load_revive_buffer(eval_data_path)
            except:
                val_buffer = None

    return train_buffer, val_buffer