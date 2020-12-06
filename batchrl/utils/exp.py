import os
import uuid
import random

import aim
import torch
import numpy as np

def setup_seed(seed=1024):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
        

def select_free_cuda():
    # 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')  # 删除临时生成的 tmp 文件

    import uuid
    return np.argmax(memory_gpu)



def init_exp_logger(experiment_name=None,flush_frequency=1):       
    aim_logger = aim.Session(experiment=experiment_name, flush_frequency=flush_frequency)

    return aim_logger