import logging
import logging.config
from ringer.constants import LOGGING_CONFIG, RANDOM_STATE
from ringer.regions import get_named_et_eta_regions
import os
import json
from itertools import product
import numpy as np
import pandas as pd
from Gaugi import load as gload
from Gaugi import save as gsave
from kepler.pandas.readers import load as kload
from sklearn.model_selection import StratifiedKFold

def load_data_with_func(filepath, load_func):
    if load_func == 'gload':
        data = gload(filepath)
        data_df = pd.DataFrame(data['data'], columns=data['features'])
        target_df = pd.DataFrame(data['target'], columns=['target'])
        data_df = pd.concat([data_df, target_df], axis=1)
    elif load_func == 'kload':
        data_df = kload(filepath)
    elif load_func == 'parquet':
        data_df = pd.read_parquet(filepath)
    else:
        raise ValueError('Available load functions are gload, kload and parquet')
    return data_df

N_FOLDS = 10
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')

# dataset = 'mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2'; load_func = 'kload'; region_name = 'L2Calo_2017'
dataset = 'data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97'; load_func = 'gload'; region_name = 'L2Calo_2017_alt'
# dataset = 'data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'; load_func = 'kload'; region_name = 'L2Calo_2017'

basepath = os.path.join('..', '..')
datapath = os.path.join(basepath, 'data', dataset)
filepath = os.path.join(datapath, dataset + '_et{et}_eta{eta}.npz')
crossval = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
out_dataset = 'new_' + dataset
out_datapath =  os.path.join(basepath, 'data', out_dataset, 'data.parquet')
out_filepath =  os.path.join(out_datapath, 'data_et{et}_eta{eta}.parquet')
if not os.path.exists(out_datapath):
    os.makedirs(out_datapath)
et_eta_regions, n_ets, n_etas = get_named_et_eta_regions(region_name)

#Reading data and generting folds
splitted_data = list()
range_start = 0
data_df = None
for region in et_eta_regions:
    logger.info(f'Processing (et_idx, eta_idx) {(region.et_idx, region.eta_idx)}')
    del data_df
    data_df = load_data_with_func(filepath.format(et=region.et_idx, eta=region.eta_idx), load_func)
    if data_df.empty:
        raise ValueError('The data is empty')
    
    if dataset == 'data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97':
        #This is tested because this dataset has a predifined kfold
        n_samples = len(data_df)
        ifold = 0.0
        for train_idx, val_idx in crossval.split(data_df.values, data_df[['target']].values):
            logger.info(f'Adding fold {ifold}')
            new_ids = np.arange(ifold, n_samples, N_FOLDS, dtype='uint64')
            data_df.loc[val_idx, 'region_id'] = new_ids
            ifold += 1
        data_df['region_id'] = data_df['region_id'].astype('uint64')
    else:
        logger.info('Adding region id')
        region_id = np.arange(len(data_df), dtype='uint64')
        np.random.shuffle(region_id)
        data_df['region_id'] = pd.Series(region_id)
    
    if data_df['region_id'].isnull().any():
        raise RuntimeError('Region id has nan values')
        
    data_df["id"] = np.arange(range_start, range_start+len(data_df))
    range_start += len(data_df)
    logger.info(f'Saving (et_idx, eta_idx) {(region.et_idx, region.eta_idx)}')
    if data_df.empty:
        raise ValueError('The save_df is empty')
    data_df.to_parquet(out_filepath.format(et=region.et_idx, eta=region.eta_idx))

logger.info('Exporting schema')
dataset_dir, table_name = os.path.split(out_datapath)
table_name = table_name.replace('.parquet', '')
with open(os.path.join(dataset_dir, f'{table_name}_schema.json'), 'w') as json_file:
    json.dump(data_df.dtypes.astype(str).to_dict(), json_file, indent=4)
logger.info('Saved')