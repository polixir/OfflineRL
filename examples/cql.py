import os
import sys
from batchrl.utils.config import parse_config
from batchrl.algo.modelfree import cql
from batchrl.config.algo import cql_config
from batchrl.data.d4rl import load_d4rl_buffer

from batchrl.evaluation.d4rl import d4rl_eval

algo = cql
algo_config = parse_config(cql_config)

algo_init = algo.algo_init(algo_config)
algo_runner = algo.AlgoTrainer(algo_init, algo_config)
offlinebuffer = load_d4rl_buffer("walker2d-medium-v0")
algo_runner.train(offlinebuffer,eval_fn=d4rl_eval)