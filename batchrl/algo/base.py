from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def train(self, 
              history_buffer,
              eval_fn=None,):
        pass
    
    def _sync_weight(self, net_target, net, soft_target_tau = 5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)
    
    def eval(self,):
        pass
    
    @abstractmethod
    def save_model(self,):
        pass
    
    @abstractmethod
    def get_model(self,):
        pass
