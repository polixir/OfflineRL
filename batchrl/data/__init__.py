import os
import time
import random
from loguru import logger

from batchrl.utils.logger import log_path
from batchrl.utils.io import create_dir, download_helper, read_json

dataset_dir = os.path.join(log_path(),"./batchrl_datasets")
create_dir(dataset_dir)


def load_data_by_task(task):
    time.sleep(random.random()*10)
    if task.startswith("d4rl"):
        pass
    else:
        from batchrl.data.revive import load_revive_buffer
        
        data_map_path = os.path.join(dataset_dir, "data_map.json")
        
        if not os.path.exists(data_map_path):
            url = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"
            data = download_helper(url, data_map_path)
            
        data_map = read_json(data_map_path)
        if task not in data_map.keys():
            url = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"
            data = download_helper(url, data_map_path)
            
            
        task_name = [i for i in data_map.keys() if i.startswith(task) and i.endswith(".npz")]
        if len(task_name) == 0:
            task_name = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir) if i.startswith(task) and i.endswith(".npz")]
         
        if len(task_name) == 0: 
            logger.info('No task dataset: {}, Pleace check the task name!', task)
        elif len(task_name) > 1:
            logger.info('Please check your task name!', task)
        else:
            data_path = os.path.join(dataset_dir, task_name[0])
               
            if not os.path.exists(data_path):
                data_url = data_map[task_name[0]]
                logger.info('Download {}', data_url)
                download_res = download_helper(data_url, data_path)
            offlinebuffer = load_revive_buffer(data_path)

            return offlinebuffer