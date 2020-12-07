import numpy as np
import torch

from tianshou.data import to_torch, to_torch_as, to_numpy
from tianshou.data import Batch

def to_array_as(x, y):    
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return to_numpy(x)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tenso):
        return to_torch_as(x)
    else:
        return x