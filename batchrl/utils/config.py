def parse_config(cfg_module):
    args = [ i for i in dir(cfg_module) if not i.startswith("__")]
    args = { arg: getattr(cfg_module, arg)  for arg in args}

    return args
    