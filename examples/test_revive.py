import fire

from batchrl.algo import algo_select
from batchrl.data.revive import load_revive_buffer
from batchrl.evaluation.d4rl import mujoco_eval_fn
from batchrl.evaluation.fqe import fqe_eval_fn

def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
     
    offlinebuffer = load_revive_buffer(algo_config["dataset_path"])

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    algo_trainer.train(offlinebuffer,callback_fn=mujoco_eval_fn(algo_config["task"], eval_episodes=10))

    
if __name__ == "__main__":
    fire.Fire(run_algo)
    