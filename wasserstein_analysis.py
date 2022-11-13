import os
import logging
import logging.config
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import mplhep as hep
# plt.style.use(hep.style.ATLAS)
# plt.rc('legend',fontsize='large')
# plt.rc('axes',labelsize='x-large')
# plt.rc('text',usetex='false')
# plt.rc('xtick', labelsize='large')
from scipy.stats import wasserstein_distance
from itertools import combinations
from ringer.constants import LOGGING_CONFIG
from ringer.data import NamedDatasetLoader, load_var_infos
from ringer.crossval import ColumnKFold
from ringer.utils import medium_keys_mapping
from typing import Mapping, Callable

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')

# Arguments
n_folds = 5
sequential_col_name = 'region_id'
out_filepath = os.path.join('..', '..', 'data', 'wass_distances.csv')
test = True


def get_shower_shapes_to_analyze(var_infos: pd.DataFrame) -> pd.Series:
    is_ss = var_infos['type'] == 'shower_shape'
    ss_to_analyze = [
        var_infos.loc[is_ss & (~var_infos['l2calo'].isnull()), 'l2calo'].rename('col'),
        var_infos.loc[is_ss & (var_infos['l2calo'].isnull()), 'offline'].rename('col')
    ]
    ss_to_analyze = pd.concat(ss_to_analyze)
    return ss_to_analyze

def get_wasserstein_distances(data: Mapping[str, pd.DataFrame],
    ss_filters: Mapping[str, Callable[[pd.Series], pd.Series]], description: str,
    ss_to_analyze: pd.Series) -> pd.DataFrame:

    data_combinations = list(combinations(data.keys(), 2))
    cols = [f'{left}_{right}' for left, right in data_combinations]
    wass_distances = pd.DataFrame(index=ss_to_analyze.index, columns=cols, dtype=float)
    wass_distances.index.name='name'
    
    for ss_name, ss_col in ss_to_analyze.iteritems():
        
        for left, right in data_combinations:
            logger.info(f'{ss_name}: computing wasserstein_distance({left}, {right}) {description}')
            left_data = ss_filters[ss_name](data[left][ss_col])
            right_data = ss_filters[ss_name](data[right][ss_col])
            wass_distances.loc[ss_name, f'{left}_{right}'] = wasserstein_distance(left_data, right_data)
    
    wass_distances['description'] = description
    return wass_distances

logger.info('Loading var_infos and getting shower shapes to analyze')
var_infos = load_var_infos()
ss_to_analyze = get_shower_shapes_to_analyze(var_infos)
load_cols = ss_to_analyze.to_list() + ['region_id', 'id']
logger.info('Reading MC16 Boosted data')
boosted_data = NamedDatasetLoader('MC16 Boosted', test).load_data_df(columns=load_cols)

logger.info('Reading 2017 Medium data')
medium_col_mapping = medium_keys_mapping(load_cols)
medium_col_mapping['target'] = 'target'
collision_data = NamedDatasetLoader('2017 Medium', test).load_data_df(columns=list(medium_col_mapping.keys()))
collision_data.rename(medium_col_mapping, axis=1, inplace=True)
jet_label = ~collision_data['target'].astype(bool)
el_label = collision_data['target'].astype(bool)
logger.info(f'There are {jet_label.sum()} jets and {el_label.sum()} electrons')
# collision_data.drop(['target'], axis=1, inplace=True)
logger.info('Loaded collision data')

# el_data = collision_data.loc[el_label]
# jet_data = collision_data.loc[jet_label]
# del collision_data
# logger.info('Loaded el and jet data')

# data = {
#     'boosted': boosted_data,
#     'el': el_data,
#     'jet': jet_data
# }

ss_filters = {
    'f3': lambda x: x,
    'weta2': lambda x: x[x < 98],
    'reta': lambda x: x,
    'wstot': lambda x: x[x != -9999],
    'eratio': lambda x: x[x < 98],
    'f1': lambda x: x,
    'rphi': lambda x: x[x.between(-0.5, 1.5, inclusive='both')],
    'rhad': lambda x: x,
    'rhad1': lambda x:x
}
cross_val = ColumnKFold(n_folds=n_folds, sequential_col_name='region_id')
collision_split = cross_val.split(collision_data)
boosted_split = cross_val.split(boosted_data)
all_wass_distances = list()
for ifold in range(n_folds):
    collision_fold_idx, _ = next(collision_split)
    boosted_fold_idx, _ = next(boosted_split)
    fold_data = {
        'boosted': boosted_data.loc[boosted_fold_idx],
        'el': collision_data.loc[collision_fold_idx & el_label],
        'jet': collision_data.loc[collision_fold_idx & jet_label]
    }
    wass_distances = get_wasserstein_distances(fold_data, ss_filters, 
        description=f'fold_{ifold}', ss_to_analyze=ss_to_analyze)
    all_wass_distances.append(wass_distances)

logger.info('Computing distances for all the data')
complete_data = {
    'boosted': boosted_data,
    'el': collision_data.loc[el_label],
    'jet': collision_data.loc[jet_label]
}
wass_distances = get_wasserstein_distances(complete_data, ss_filters, 
        description='complete', ss_to_analyze=ss_to_analyze)
logger.info('Computed all')
all_wass_distances.append(wass_distances)
all_wass_distances_df = pd.concat(all_wass_distances, axis=0)
all_wass_distances_df = all_wass_distances_df.reset_index()
all_wass_distances_df.to_csv(out_filepath)