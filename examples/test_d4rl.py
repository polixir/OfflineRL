import fire

from batchrl.algo import algo_select
from batchrl.data.d4rl import load_d4rl_buffer
from batchrl.evaluation.d4rl import d4rl_eval_fn
from batchrl.evaluation.fqe import fqe_eval_fn

def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
     
    offlinebuffer = load_d4rl_buffer(algo_config["task"], 
                                     algo_config["episode_num"] if "episode_num" in algo_config.keys() else None, 
                                     algo_config["step_num"] if "step_num" in algo_config.keys() else None)

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    algo_trainer.train(offlinebuffer,callback_fn=d4rl_eval_fn(algo_config["task"], eval_episodes=100))
    #algo_trainer.train(offlinebuffer,callback_fn=fqe_eval_fn())

    
if __name__ == "__main__":
    fire.Fire(run_algo)
    