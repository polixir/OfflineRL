from loguru import logger
import warnings

warnings.filterwarnings('ignore')


from batchrl.config.algo import cql_config, plas_config, mopo_config, moose_config, mlas_config, bcqd_config, bcq_config, bc_config, crr_config
from batchrl.utils.config import parse_config
from batchrl.algo.modelfree import cql, plas, mlas, bcqd, bcq, bc, crr
from batchrl.algo.modelbase import mopo, moose

algo_dict = {
    'bc' : {"algo" : bc, "config" : bc_config},
    'bcq' : {"algo" : bcq, "config" : bcq_config},
    'bcqd' : {"algo" : bcqd, "config" : bcqd_config},
    "cql" : {"algo" : cql, "config" : cql_config},
    "crr" : {"algo" : crr, "config" : crr_config},
    "plas" : {"algo" : plas, "config" : plas_config},
    "mlas" : {"algo" : mlas, "config" : mlas_config},
    'moose' : {"algo" : moose, "config" : moose_config},
    'mopo': {"algo" : mopo, "config": mopo_config},
}

def algo_select(command_args, algo_config_module=None):
    algo_name = command_args["algo_name"]
    logger.info('Use {} algorithm!', algo_name)
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]
    
    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)
    algo_config.update(command_args)
    
    algo_init = algo.algo_init
    algo_trainer = algo.AlgoTrainer
    
    return algo_init, algo_trainer, algo_config
    
    