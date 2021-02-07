import fire

from offlinerl.algo import algo_select
from offlinerl.data import load_data_from_neorl
from offlinerl.evaluation import get_defalut_callback, OnlineCallBackFunction


def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    train_buffer, val_buffer = load_data_from_neorl(algo_config["task"], algo_config["task_data_type"], algo_config["task_train_num"])
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = OnlineCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"])

    algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)

if __name__ == "__main__":
    fire.Fire(run_algo)
    