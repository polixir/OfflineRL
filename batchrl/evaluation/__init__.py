from abc import ABC, abstractmethod


class CallBack():
    def __init__(self,
                 actor, 
                 train_buffer, 
                 callback_type="env",
                 eval_buffer=None,
                 args=None,
                 **kwargs):
        self.actor = actor
        self.train_buffer = train_buffer
        self.eval_buffer_buffer = eval_buffer
        self.args = args
        self.kwargs = kwargs
        
        self(callback_type)
    
    def __call__(self, callback_type):
        if callback_type == "env":
            pass
        elif callback_type == "fqe":
            pass
        elif callback_type == "agmodel":
            pass
        else:
            raise NotImplementedError
            
        