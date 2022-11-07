import os
import pandas as pd
from typing import List
from ringer.constants import NAMED_DATASETS


class LocalDataLoader(object):

    def __init__(self, dataset_path: str):

        dataset_path = str(dataset_path)
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            raise FileNotFoundError(f'{datset_path} does not exist.')
    
    def load_data_df(self, columns: List[str]=None):
        data_path = os.path.join(self.dataset_path, 'data.parquet')
        data_df = pd.read_parquet(data_path, columns=columns)
        return data_df

class NamedDatasetLoader(LocalDataLoader):
    
    def __init__(self, dataset_name: str):
        self.dataset_name = str(dataset_name)
        self.dataset_path = NAMED_DATASETS[self.dataset_name]