import os
import pandas as pd
from typing import List, Union
from ringer.constants import NAMED_DATASETS, VAR_INFOS_PATH, VAR_INFOS_DTYPES

class LocalDataLoader(object):


    def __init__(self, dataset_path: str, test:bool=False):

        dataset_path = str(dataset_path)
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            raise FileNotFoundError(f'{dataset_path} does not exist.')
        self.test = bool(test)


    def get_df_path(self, base_name: str) -> str:
        dirname = f"{base_name}.parquet"
        df_path = os.path.join(self.dataset_path, dirname)
        if self.test:
            filename = f"{base_name}_et4_eta0.parquet"
            df_path = os.path.join(df_path, filename)
        return df_path


    def load_data_df(self, columns: Union[List[str], None] = None) -> pd.DataFrame:
        data_df_path = self.get_df_path("data")
        data_df = pd.read_parquet(data_df_path, columns=columns)
        return data_df


    def load_ringer_df(self, ringer_version: str, columns: Union[List[str], None] = None) -> pd.DataFrame:
        base_name = f"ringer_v{ringer_version}"
        ringer_df_path = self.get_df_path(base_name)
        ringer_df = pd.read_parquet(ringer_df_path, columns=columns)
        return ringer_df


    def save_data_df(self, data_df: pd.DataFrame, et_bin_idx: int, eta_bin_idx: int):
        filename = f"data_et{et_bin_idx}_eta{eta_bin_idx}.parquet"
        df_path = os.path.join(self.dataset_path, 'data.parquet', filename)
        data_df.to_parquet(df_path)
    
    def save_data_df(self, ringer_df: pd.DataFrame, et_bin_idx: int, eta_bin_idx: int,
                     ringer_version: str):
        base_name = f"ringer_v{ringer_version}"
        dirname = f"{base_name}.parquet"
        filename = f"{base_name}_et{et_bin_idx}_eta{eta_bin_idx}.parquet"
        df_path = os.path.join(self.dataset_path, dirname, filename)
        ringer_df.to_parquet(df_path)


class NamedDatasetLoader(LocalDataLoader):

    def __init__(self, dataset_name: str, test:bool=False):
        self.dataset_name = str(dataset_name)
        super().__init__(NAMED_DATASETS[self.dataset_name], test)


def load_var_infos(var_infos_path:Union[str, None]=None):
    if var_infos_path is None:
        var_infos_path = VAR_INFOS_PATH
    var_infos = pd.read_csv(var_infos_path,
                            index_col=0, dtype=VAR_INFOS_DTYPES)
    return var_infos

def get_electron_label(data: pd.DataFrame, criterion: str):
    return (data['target'] == 1) & (data[criterion] == 1)

def get_jet_label(data: pd.DataFrame, criterion: str):
    return (data['target'] != 1) & (data[criterion] != 1)\

LABEL_UTILITIES = {
    'electron': get_electron_label,
    'jet': get_jet_label
}