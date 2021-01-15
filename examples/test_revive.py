import fire

from batchrl.algo import algo_select
from batchrl.data import load_data_by_task
from batchrl.data.revive import load_revive_buffer
from batchrl.evaluation import get_defalut_callback


def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)

    train_buffer, val_buffer = load_data_by_task(algo_config["task"])

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = get_defalut_callback()
    callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"])

    algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)

if __name__ == "__main__":
    fire.Fire(run_algo)
    