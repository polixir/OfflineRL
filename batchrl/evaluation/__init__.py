from abc import ABC, abstractmethod


class CallBackFn():
    def __init__(self,):
        self.index = 0
    
    @abstractmethod
    def __call__(self, actor, buffer, **kwargs):
        pass
        