import os
import pandas as pd
from typing import List


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
        