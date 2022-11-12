import os
import pandas as pd
from typing import List, Union
from ringer.constants import NAMED_DATASETS, VAR_INFOS_PATH, VAR_INFOS_DTYPES

class LocalDataLoader(object):

    def __init__(self, dataset_path: str):

        dataset_path = str(dataset_path)
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            raise FileNotFoundError(f'{dataset_path} does not exist.')

    def load_data_df(self, columns: Union[List[str], None] = None):
        data_path = os.path.join(self.dataset_path, 'data.parquet')
        data_df = pd.read_parquet(data_path, columns=columns)
        return data_df


class NamedDatasetLoader(LocalDataLoader):

    def __init__(self, dataset_name: str):
        self.dataset_name = str(dataset_name)
        self.dataset_path = NAMED_DATASETS[self.dataset_name]


def load_var_infos(var_infos_path:str=None):
    if var_infos_path is None:
        var_infos_path = VAR_INFOS_PATH
    var_infos = pd.read_csv(var_infos_path,
                            index_col=0, dtype=VAR_INFOS_DTYPES)
    var_infos = var_infos.set_index('name')
    return var_infos

def get_electron_label(data: pd.DataFrame, criterion: str):
    return (data['target'] == 1) & (data[criterion] == 1)

def get_jet_label(data: pd.DataFrame, criterion: str):
    return (data['target'] != 1) & (data[criterion] != 1)\

LABEL_UTILITIES = {
    'electron': get_electron_label,
    'jet': get_jet_label
}