import os
import sys
from batchrl.utils.config import parse_config
from batchrl.algo.offline import cql
from batchrl.data.d4rl import load_d4rl_buffer
from batchrl.trainer.offline import OfflineTrainer

algo = cql.algo
algo_config = parse_config(cql.config)

init = algo.algo_init(algo_config)
offlinebuffer = load_d4rl_buffer("walker2d-medium-v0")

algo_runner = algo.AlgoTrainer(init,algo_config)
trainer = OfflineTrainer(algo_runner, offlinebuffer, algo_config)
trainer.train()