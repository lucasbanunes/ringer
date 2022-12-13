import os
import logging
import logging.config
import pandas as pd
from itertools import product
from ringer.constants import LOGGING_CONFIG
from ringer.data import NamedDatasetLoader, load_var_infos
from ringer.crossval import ColumnKFold
from ringer.utils import medium_keys_mapping
from ringer.scalers import MinMaxScaler
from ringer.infgeometry import wasserstein_distance
from collections import defaultdict

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')
FOLD_TYPES = ['train', 'test']

# Arguments
n_folds = 10
sequential_col_name = 'region_id'
output_dir = os.path.join('..', '..', 'data')
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


def build_data_values_from_scaler(scaler: MinMaxScaler,
                                  ifold: int, fold_type: str
                                  ) -> pd.DataFrame:
    frmt_min_values = scaler.min_values_ \
        .reset_index(names='description') \
        .melt(id_vars='description', var_name='cols')
    frmt_min_values['description'] += f'_fold_{ifold}_{fold_type}'
    frmt_min_values['name'] = 'min'

    frmt_max_values = scaler.max_values_ \
        .reset_index(names='description') \
        .melt(id_vars='description', var_name='cols')
    frmt_max_values['description'] += f'_fold_{ifold}_{fold_type}'
    frmt_max_values['name'] = 'max'

    data_values = pd.concat([frmt_min_values, frmt_max_values], axis=0)
    correct_ordering = [
        'name', 'cols', 'value', 'description'
    ]
    data_values = data_values[correct_ordering]

    return data_values


logger.info('Loading var_infos and getting shower shapes to analyze')
var_infos = load_var_infos()
ss_to_analyze = get_shower_shapes_to_analyze(var_infos)
if test:
    ss_to_analyze = ss_to_analyze.iloc[:3]
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
all_data_values = list()
scalers = defaultdict(dict)

for var_name, var_col in ss_to_analyze.items():
    for ifold, fold_type in product(range(n_folds), FOLD_TYPES):
        logger.info("Computing distances "
                    f"for {var_name} in fold {ifold} {fold_type}")

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
            data_values = build_data_values_from_scaler(
                scaler, ifold, fold_type
            )
            all_data_values.append(data_values)
            scaled_data = scaler.transform(fold_data)
        else:
            scaled_data = scaler.transform(fold_data)
        wass_distances = wasserstein_distance(scaled_data)
        wass_distances = wass_distances.T
        wass_distances['description'] = f'fold_{ifold}_{fold_type}'
        all_wass_distances.append(wass_distances)

    del fold_data


logger.info('Computed all')
all_wass_distances_df = pd.concat(all_wass_distances, axis=0)
all_wass_distances_df = all_wass_distances_df.reset_index()
all_wass_distances_df.to_csv(os.path.join(output_dir, 'wass_distances.csv'))

all_data_values_df = pd.concat(all_data_values, axis=0)
all_data_values_df = all_data_values_df.reset_index()
all_data_values_df.to_csv(os.path.join(output_dir, 'data_values.csv'))
