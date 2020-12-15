import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tianshou.data import Batch
from tianshou.data import to_torch, to_torch_as, to_numpy


def to_array_as(x, y):    
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return to_numpy(x)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tenso):
        return to_torch_as(x)
    else:
        return x
    
class SampleBatch(Batch):
    def sample(self,batch_size):
        length = len(self)
        assert 1 <= batch_size
        
        indices = np.random.randint(0, length, batch_size)
        
        return self[indices]

def sample(batch : Batch, batch_size : int):
    length = len(batch)
    assert 1 <= batch_size
    
    indices = np.random.randint(0, length, batch_size)

    return batch[indices]


def get_scaler(data):
    scaler = MinMaxScaler((-1,1))
    scaler.fit(data)
    
    return scaler
