from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def train(self, 
              history_buffer,
              eval_fn=None,):
        pass
    
    def eval(self,):
        pass
