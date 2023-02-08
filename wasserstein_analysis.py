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
from datetime import datetime

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')
FOLD_TYPES = ['train', 'test']

# Arguments
n_folds = 10
sequential_col_name = 'region_id'
script_datetime = datetime.now()
script_time_str = script_datetime.strftime("%Y_%m_%d_%H_%M_%S")
output_dir = os.path.join('..', '..', 'data', f"{script_time_str}_wasserstein_analysis")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
test = False
vars2analyze = [
    "trig_L2_cl_reta",
    "trig_L2_cl_eratio",
    "trig_L2_cl_f1",
    "trig_L2_cl_f3",
    "trig_L2_cl_wstot",
    "trig_L2_cl_weta2",
    "el_rhad",
    "el_rhad1",
    "el_rphi"
]


def get_shower_shapes_to_analyze(var_infos: pd.DataFrame, vars2analyze) -> pd.Series:
    ss_to_analyze = var_infos.loc[vars2analyze, "name"]
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
# ss_to_analyze = get_shower_shapes_to_analyze(var_infos, vars2analyze)
# logger.debug(f"ss_to_analyze: {ss_to_analyze}")
load_cols = vars2analyze + ['region_id', 'id']
logger.info('Reading MC16 Boosted data')
logger.debug(f"Load cols: {load_cols}")
boosted_data = NamedDatasetLoader('mc16_boosted', test) \
    .load_data_df(columns=load_cols)

logger.info('Reading 2017 Medium data')
medium_col_mapping = medium_keys_mapping(load_cols)
medium_col_mapping['target'] = 'target'
logger.debug(f"Load cols: {medium_col_mapping.keys()}")
collision_data = NamedDatasetLoader('2017_medium', test) \
    .load_data_df(columns=list(medium_col_mapping.keys()))
collision_data.rename(medium_col_mapping, axis=1, inplace=True)
jet_label = ~collision_data['target'].astype(bool)
el_label = collision_data['target'].astype(bool)
logger.info(f'There are {jet_label.sum()} jets and {el_label.sum()} electrons')
logger.info('Loaded collision data')


ss_filters = {
    'trig_L2_cl_f3': lambda x: x,
    'trig_L2_cl_weta2': lambda x: x[x < 98],
    'trig_L2_cl_reta': lambda x: x,
    'trig_L2_cl_wstot': lambda x: x[x != -9999],
    'trig_L2_cl_eratio': lambda x: x[x < 98],
    'trig_L2_cl_f1': lambda x: x,
    'el_rphi': lambda x: x[x.between(-0.5, 1.5, inclusive='both')],
    'el_rhad': lambda x: x,
    'el_rhad1': lambda x: x
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

for var_name in vars2analyze:
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
            'boosted': filter_func(boosted_fold[var_name]).to_frame(),
            'el': filter_func(el_fold[var_name]).to_frame(),
            'jet': filter_func(jet_fold[var_name]).to_frame()
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
