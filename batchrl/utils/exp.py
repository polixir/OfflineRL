import os
import uuid
import random

import aim
import torch
import numpy as np
from loguru import logger

from batchrl.utils.logger import log_path

def setup_seed(seed=1024):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def select_free_cuda():
    # 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
    tmp_name = str(uuid.uuid1()).replace("-","")
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >'+tmp_name)
    memory_gpu = [int(x.split()[2]) for x in open(tmp_name, 'r').readlines()]
    os.system('rm '+tmp_name)  # 删除临时生成的 tmp 文件
    
    return np.argmax(memory_gpu)

def set_free_device_fn():
    device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'

    return device


def init_exp_logger(repo=None, experiment_name=None, flush_frequency=1):
    if repo is None:
        repo = os.path.join(log_path(),"./.aim")
        if not os.path.exists(repo):
            logger.info('{} dir is not exist, create {}',repo, repo)
            os.system(str("cd " + os.path.join(repo,"../") + "&& aim init"))
    
    aim_logger = aim.Session(repo = repo, experiment=experiment_name, flush_frequency=flush_frequency)
    aim_logger.experiment_name = experiment_name
    
    return aim_logger