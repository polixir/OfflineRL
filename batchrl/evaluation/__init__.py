import ray

from copy import deepcopy
from abc import ABC, abstractmethod
from collections import OrderedDict

from batchrl.utils.env import get_env
from batchrl.evaluation.porl import test_on_real_env
from batchrl.evaluation.fqe import fqe_eval_fn

def EvaluationCallBackFn():
    return EvaluationCallBack()


class EvaluationCallBack:
    def __init__(self):
        self.epoch = 0

    def __call__(self,
                 policy, 
                 train_buffer, 
                 callback_type = ["env", "fqe"],
                 env = None,
                 val_buffer = None,
                 args = None,
                 **kwargs):

        self.epoch += 1
        buffer = val_buffer or train_buffer
        
        if isinstance(callback_type, str):
            callback_type_list = [callback_type,]
        else:
            callback_type_list = callback_type

        policy = deepcopy(policy).cpu()

        eval_res = OrderedDict()
        if "env" in callback_type_list:
            if env is None:
                try:
                    env = get_env(args["task"])
                except Exception  as e:
                    raise NotImplementedError("Please check the env for evaluation.")
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            eval_res.update(test_on_real_env(policy, env))
        if self.epoch % 25 == 0:
            if "fqe" in callback_type_list:
                eval_res.update(fqe_eval_fn()(policy, buffer))
            if "ope" in callback_type_list:
                pass
        for callback_fn in callback_type_list:
            if not isinstance(callback_fn, str):
                eval_res.update(callback_fn(policy, 
                                            train_buffer,
                                            env,
                                            buffer,
                                            args,
                                            **kwargs))

        return eval_res