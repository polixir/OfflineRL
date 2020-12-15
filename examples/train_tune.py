import fire
from ray import tune

from batchrl.algo import algo_select
from batchrl.data.revive import load_revive_buffer
from batchrl.evaluation.gym import gym_policy_eval,gym_env_eval
from batchrl.evaluation.fqe import fqe_eval_fn

def training_function(config):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])
    offlinebuffer = load_revive_buffer(algo_config["dataset_path"])
    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    score = algo_trainer.train(offlinebuffer,callback_fn=gym_policy_eval(algo_config["task"], eval_episodes=10))
    
    return score

def run_algo(**kwargs):
    config = {}
    config["kwargs"] = kwargs
    _, _, algo_config = algo_select(kwargs)
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
    