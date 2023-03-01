import os
import json
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


    def get_df_path(self, base_name: str, et_bin_idx=None, eta_bin_idx=None) -> str:
        dirname = f"{base_name}.parquet"
        df_path = os.path.join(self.dataset_path, dirname)
        has_idx = (et_bin_idx is not None) and (eta_bin_idx is not None)
        if self.test:
            filename = f"{base_name}_et4_eta4.parquet"
            df_path = os.path.join(df_path, filename)
        elif has_idx:
            filename = f"{base_name}_et{et_bin_idx}_eta{eta_bin_idx}.parquet"
            df_path = os.path.join(df_path, filename)
            
        return df_path


    def load_data_df(self, columns: Union[List[str], None] = None,
                     et_bin_idx=None, eta_bin_idx=None) -> pd.DataFrame:
        data_df_path = self.get_df_path("data", et_bin_idx, eta_bin_idx)
        data_df = pd.read_parquet(data_df_path, columns=columns)
        return data_df


    def load_strategy_df(self, strategy_name: str, columns: Union[List[str], None] = None,
                       et_bin_idx=None, eta_bin_idx=None) -> pd.DataFrame:
        strategy_df_path = self.get_df_path(strategy_name, et_bin_idx, eta_bin_idx)
        strategy_df = pd.read_parquet(strategy_df_path, columns=columns)
        return strategy_df

    def dump_data_df(self, data_df: pd.DataFrame, et_bin_idx: int, eta_bin_idx: int):
        base_name = "data"
        try:
            data_schema = self.load_schema(base_name)
        except FileNotFoundError:
            self.dump_schema(base_name, data_df)
        filename = f"data_et{et_bin_idx}_eta{eta_bin_idx}.parquet"
        dirpath = os.path.join(self.dataset_path, 'data.parquet')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        df_path = os.path.join(dirpath, filename)
        data_df.to_parquet(df_path)
    
    def dump_strategy_df(self, strategy_df: pd.DataFrame, strategy_name: str,
                       et_bin_idx: int, eta_bin_idx: int):
        try:
            data_schema = self.load_schema(strategy_name)
        except FileNotFoundError:
            self.dump_schema(strategy_name, strategy_df)
        dirname = f"{strategy_name}.parquet"
        filename = f"{strategy_name}_et{et_bin_idx}_eta{eta_bin_idx}.parquet"
        dirpath = os.path.join(self.dataset_path, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        df_path = os.path.join(dirpath, filename)
        strategy_df.to_parquet(df_path)
    
    def get_data_df_dtypes(self):
        data_schema = self.load_schema("data")
        data_df_dtypes = pd.Series(data_schema, name="data_df_dtypes")
        return data_df_dtypes
    
    def get_strategy_df_dtypes(self, strategy_name: str):
        data_schema = self.load_schema(strategy_name)
        strategy_df_dtypes = pd.Series(data_schema, name=f"{strategy_name}_df_dtypes")
        return strategy_df_dtypes
    
    def load_schema(self, base_name):
        schema_path = os.path.join(self.dataset_path, f"{base_name}_schema.json")
        with open(schema_path, "r") as json_file:
            data_schema = json.load(json_file)
        return data_schema
    
    def dump_schema(self, base_name, df):
        schema_path = os.path.join(self.dataset_path, f"{base_name}_schema.json")
        data_schema = {key: str(value) for key, value in df.dtypes.items()}
        with open(schema_path, "w") as json_file:
            json.dump(data_schema, json_file, indent=4)


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