import configparser

def parse_config(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    
    sections = cfg.sections()
    
    for section in sections:
        options = cfg.options(section)
        print(options)
    