import uuid
import aim

class exp_logger():
    def __init__(self, experiment_name=None,flush_frequency=1):
        if experiment_name is None:
            experiment_name = str(uuid.uuid1())
            
        self.aim_logger = aim.Session(experiment=experiment_name, flush_frequency=flush_frequency)
    
    def log_hparams(self, hparams_dict):
        self.aim_logger.set_params(hparams_dict, name='hparams')
        
    #def log_metric(self, **kwargs):
    #    self.aim_logger.trick()