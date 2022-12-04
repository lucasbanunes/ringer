import os
import logging
import logging.config
import pandas as pd
from scipy.stats import wasserstein_distance
from itertools import combinations, product
from ringer.constants import LOGGING_CONFIG
from ringer.data import NamedDatasetLoader, load_var_infos
from ringer.crossval import ColumnKFold
from ringer.utils import medium_keys_mapping
from ringer.scalers import MinMaxScaler
from ringer.infgeometry import wasserstein_distance
from typing import Mapping, Callable

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')
FOLD_TYPES = ['train', 'test']

# Arguments
n_folds = 10
sequential_col_name = 'region_id'
out_filepath = os.path.join('..', '..', 'data', 'wass_distances.csv')
test = True


def get_shower_shapes_to_analyze(var_infos: pd.DataFrame) -> pd.Series:
    is_ss = var_infos['type'] == 'shower_shape'
    ss_to_analyze = [
        var_infos.loc[is_ss & (~var_infos['l2calo'].isnull()), 'l2calo']
        .rename('col'),
        var_infos.loc[is_ss & (var_infos['l2calo'].isnull()), 'offline']
        .rename('col')
    ]
    ss_to_analyze = pd.concat(ss_to_analyze)
    return ss_to_analyze


logger.info('Loading var_infos and getting shower shapes to analyze')
var_infos = load_var_infos()
ss_to_analyze = get_shower_shapes_to_analyze(var_infos)
load_cols = ss_to_analyze.to_list() + ['region_id', 'id']
logger.info('Reading MC16 Boosted data')
boosted_data = NamedDatasetLoader('mc16_boosted', test) \
    .load_data_df(columns=load_cols)

logger.info('Reading 2017 Medium data')
medium_col_mapping = medium_keys_mapping(load_cols)
medium_col_mapping['target'] = 'target'
collision_data = NamedDatasetLoader('2017_medium', test) \
    .load_data_df(columns=list(medium_col_mapping.keys()))
collision_data.rename(medium_col_mapping, axis=1, inplace=True)
jet_label = ~collision_data['target'].astype(bool)
el_label = collision_data['target'].astype(bool)
logger.info(f'There are {jet_label.sum()} jets and {el_label.sum()} electrons')
logger.info('Loaded collision data')


ss_filters = {
    'f3': lambda x: x,
    'weta2': lambda x: x[x < 98],
    'reta': lambda x: x,
    'wstot': lambda x: x[x != -9999],
    'eratio': lambda x: x[x < 98],
    'f1': lambda x: x,
    'rphi': lambda x: x[x.between(-0.5, 1.5, inclusive='both')],
    'rhad': lambda x: x,
    'rhad1': lambda x: x
}
cross_val = ColumnKFold(n_folds=n_folds, sequential_col_name='region_id')
collision_split = cross_val.split(collision_data)
collision_fold_idxs = {
    f'fold_{i}': fold_idxs
    for i, fold_idxs in enumerate(collision_split)
}
boosted_split = cross_val.split(boosted_data)
all_wass_distances = list()

for var_name, var_col in ss_to_analyze.items():
    for ifold, fold_type in product(range(n_folds), FOLD_TYPES):
        logger.info("Computing distances "
                    f"for {var_name} in fold {fold_type} {ifold}")

        if fold_type == 'train':
            boosted_fold = cross_val.get_train_idx(boosted_data, ifold)
            collision_fold = cross_val.get_train_idx(collision_data, ifold)
        else:
            boosted_fold = cross_val.get_test_idx(boosted_data, ifold)
            collision_fold = cross_val.get_test_idx(collision_data, ifold)

        boosted_fold = boosted_data.loc[boosted_fold]
        el_fold = collision_data.loc[collision_fold & el_label]
        jet_fold = collision_data.loc[collision_fold & jet_label]
        filter_func = ss_filters[var_name]
        fold_data = {
            'boosted': filter_func(boosted_fold[var_col]).to_frame(),
            'el': filter_func(el_fold[var_col]).to_frame(),
            'jet': filter_func(jet_fold[var_col]).to_frame()
        }
        if fold_type == 'train':
            scaler = MinMaxScaler().fit(fold_data)
            scaled_data = scaler.transform(fold_data)
        else:
            scaled_data = scaler.transform(fold_data)
        wass_distances = wasserstein_distance(scaled_data)
        wass_distances = wass_distances.T
        wass_distances['description'] = f'fold_{ifold}_{fold_type}'
        all_wass_distances.append(wass_distances)

    del fold_data

# logger.info('Computing distances for all the data')
# complete_data = {
#     'boosted': boosted_data,
#     'el': collision_data.loc[el_label],
#     'jet': collision_data.loc[jet_label]
# }
# scaler = MinMaxScaler().fit(complete_data)
# scaled_data = scaler.transform(complete_data)
# wass_distances = get_wasserstein_distances(
#         complete_data,
#         ss_filters,
#         description='complete',
#         ss_to_analyze=ss_to_analyze
# )
logger.info('Computed all')
# all_wass_distances.append(wass_distances)
all_wass_distances_df = pd.concat(all_wass_distances, axis=0)
all_wass_distances_df = all_wass_distances_df.reset_index()
all_wass_distances_df.to_csv(out_filepath)
