import sys
from loguru import logger

from offlinerl import algo, data, evaluation, utils, config

logger_config = {
    "handlers": [
        {"sink": sys.stdout, 
         "colorize" : True, 
         #"format" : "<green>{time}</green> <level>{message}</level>",
         "format" : "<green>{time:YYYY-MM-DD at HH:mm:ss.SSS}</green> | <blue>{level}</blue> | {message}",
         "enqueue" : True,
         "backtrace" : True, 
         "diagnose" : True,
        },
    ],

}
logger.configure(**logger_config)

#logger.disable("offlinerl")
logger.enable("offlinerl")

__version__ = "0.0.1"

__all__ = [
    "algo",
    "data",
    "evaluation",
    "utils",
    "config",
]