import fire
import random
from ray import tune

from offlinerl.algo import algo_select
from offlinerl.data import load_data_from_neorl
from offlinerl.evaluation import get_defalut_callback, OnlineCallBackFunction

def training_function(config):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])
    train_buffer, val_buffer = load_data_from_neorl(algo_config["task"], algo_config["task_data_type"], algo_config["task_train_num"])
    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = OnlineCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"])

    score = algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)
    
    return score

def run_algo(**kwargs):
    config = {}
    config["kwargs"] = kwargs
    config["kwargs"]['seed'] = random.randint(0, 1000000)
    _, _, algo_config = algo_select(kwargs)
    # Prepare Dataset
    load_data_from_neorl(algo_config["task"], algo_config["task_data_type"], algo_config["task_train_num"])
    grid_tune = algo_config["grid_tune"]
    for k,v in grid_tune.items():
        config[k] = tune.grid_search(v)
    
    analysis = tune.run(
        training_function,
        config=config,
        resources_per_trial={"gpu": 1},
        queue_trials = True,
        )

    
if __name__ == "__main__":
    fire.Fire(run_algo)