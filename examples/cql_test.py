import os
import sys
from batchrl.utils.config import parse_config
from batchrl.algo.modelfree import cql
from batchrl.config.algo import cql_config
from batchrl.data.d4rl import load_d4rl_buffer
from batchrl.utils.exp import setup_seed

from batchrl.evaluation.d4rl import d4rl_eval

setup_seed(112)
algo = cql
algo_config = parse_config(cql_config)

offlinebuffer = load_d4rl_buffer(algo_config["task"])
algo_init = algo.algo_init(algo_config)
algo_runner = algo.AlgoTrainer(algo_init, algo_config)

algo_runner.train(offlinebuffer,eval_fn=d4rl_eval)