import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tianshou.data import Batch
from tianshou.data import to_torch, to_torch_as, to_numpy
from torch.utils.data import dataset
from torch.utils.data import dataloader


def to_array_as(x, y):    
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return to_numpy(x)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tensor):
        return to_torch_as(x)
    else:
        return x
    
class BufferDataset(dataset.Dataset):
    def __init__(self, buffer, batch_size=256):
        self.buffer = buffer
        self.batch_size = batch_size
        self.length = len(self.buffer)
        
    def __getitem__(self, index):
        indices = np.random.randint(0, self.length, self.batch_size)
        data = self.buffer[indices]
        
        return data
        
    def __len__(self):
        return self.length
    
    
class BufferDataloader(dataloader.DataLoader):        
    def sample(self, batch_size=None): 
        if not hasattr(self, 'buffer_loader') or batch_size != self.buffer_loader._dataset.batch_size:
            if not hasattr(self, 'buffer_loader'):
                self.buffer_loader = self.__iter__()
            elif batch_size is None:
                pass
            else:
                self.dataset.batch_size = batch_size
                self.buffer_loader = self.__iter__()
        try:
            return self.buffer_loader.__next__()
        except:
            self.buffer_loader = self.__iter__()
            return self.buffer_loader.__next__()
    
class SampleBatch(Batch):
    def sample(self, batch_size):
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
