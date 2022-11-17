import os
import logging
import logging.config
import pandas as pd
import numpy as np
from ringer.data import NamedDatasetLoader
from ringer.constants import LOGGING_CONFIG, RING_COL_NAME
from ringer.crossval import ColumnKFold

def get_descriptions(boosted_data: pd.DataFrame, collision_data: pd.DataFrame, all_data: pd.DataFrame, cols: list):
    descriptions = dict()
    collision_desc = collision_data.loc[:, cols].describe()
    descriptions['collisions'] = collision_desc
    logger.info('Computed collisions')

    el_desc =  collision_data.loc[collision_data['target'] == 1, cols].describe()
    descriptions['el'] = el_desc
    logger.info('Computed el')

    jet_desc =  collision_data.loc[collision_data['target'] == 0, cols].describe()
    descriptions['jet'] = jet_desc
    logger.info('Computed jets')

    boosted_desc = boosted_data.loc[:, cols].describe()
    descriptions['boosted'] = boosted_desc
    logger.info('Computed boosted')
    
    all_desc = all_data.loc[:, cols].describe()
    descriptions['all'] = all_desc
    logger.info('Computed all')
    
    return descriptions

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')

descriptions = dict()
data_loader = NamedDatasetLoader('2017 Medium')
boosted_loader = NamedDatasetLoader('MC16 Boosted')
trig_ring_cols = [RING_COL_NAME.format(ring_num=i) for i in range(100)]
medium_load_cols = [f'L2Calo_ring_{i}' for i in range(100)]
col_rename_map = dict(zip(medium_load_cols, trig_ring_cols))
batch_size = 5
n_folds = 5
output_path = os.path.join('analysis', 'mean_analysis')
file_name = '{data_type}_fold{ifold}_description.csv' 
if not os.path.exists(output_path):
    os.makedirs(output_path)

logger.info('Loading collisions')
collision_data = data_loader.load_data_df(medium_load_cols + ['target', 'region_id'])
collision_data = collision_data.rename(col_rename_map, axis=1)

logger.info(f'Loading boosted')
boosted_data = boosted_loader.load_data_df(trig_ring_cols + ['region_id'])

logger.info(f'Generating all')
all_data = pd.concat([collision_data, boosted_data], axis=0).drop(['target'], axis=1)

cross_val = ColumnKFold(n_folds=n_folds, sequential_col_name='region_id')
boosted_split = cross_val.split(boosted_data)
collision_split = cross_val.split(collision_data)
all_split = cross_val.split(all_data)

for ifold in range(n_folds):
    logger.info(f'Fold {ifold}')
    boosted_idx, _ = next(boosted_split)
    collision_idx, _ = next(collision_split)
    all_idx, _ = next(all_split)
    logger.info('Getting descriptions')
    descriptions = get_descriptions(
        boosted_data.loc[boosted_idx],
        collision_data.loc[collision_idx],
        all_data.loc[all_idx],
        cols=trig_ring_cols)
    
    for data_type, description in descriptions.items():
        logger.info({f'Fold {ifold} saving {data_type}'})
        file_path = os.path.join(
            output_path,
            file_name.format(data_type=data_type, ifold=ifold))
        description.to_csv(file_path)
