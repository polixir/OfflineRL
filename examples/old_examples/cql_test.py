from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import cql
from offlinerl.config.algo import cql_config
from offlinerl.data.d4rl import load_d4rl_buffer
from offlinerl.utils.exp import setup_seed

from offlinerl.evaluation.d4rl import d4rl_eval_fn

algo = cql
algo_config = parse_config(cql_config)

offlinebuffer = load_d4rl_buffer(algo_config["task"])

algo_init = algo.algo_init(algo_config)
algo_runner = algo.AlgoTrainer(algo_init, algo_config)
algo_runner.train(offlinebuffer,callback_fn=d4rl_eval_fn(algo_config["task"], eval_episodes=100))