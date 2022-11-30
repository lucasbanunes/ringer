import os
import json
from typing import List, Union, Dict, Any

class TrainingJob(object):

    def __init__(
        self,
        model_config: Union[str, Dict[str, Any]],
        model_weights: List,
        gpu: int,
        experiment: str):

        self.model_config = self.__parse_model_config(model_config)
        self.model_weights = model_weights
        self.gpu = int(gpu)
        self.experiment = str(experiment)
        self.model = None
    
    def __parse_model_config(self, model_config):
        if type(model_config) is str:
            if os.path.exists(model_config):
                with open(model_config, 'r') as json_file:
                    res = json.load(json_file)
            else:
                res = json.loads(model_config)
        else:
            res = model_config
        
        return res

    def run(self, compile_kwargs, fit_kwargs):

        self.model.compile(**compile_kwargs)
        self.model.fit(**fit_kwargs)