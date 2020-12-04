from batchrl.algo.modelfree import plas
from batchrl.config.algo import plas_config

from batchrl.utils.config import parse_config
from batchrl.data.d4rl import load_d4rl_buffer
from batchrl.evaluation.d4rl import d4rl_eval_fn


algo = plas
algo_config = parse_config(plas_config)
# 初始化
algo_init = algo.algo_init(algo_config)
# 加载数据集
offlinebuffer = load_d4rl_buffer(algo_config["task"])
# 训练
algo_runner = algo.AlgoTrainer(algo_init, algo_config)
algo_runner.train(offlinebuffer,callback_fn=d4rl_eval_fn(algo_config["task"], eval_episodes=1))