import logging
import logging.config
from packages.constants import LOGGING_CONFIG, RANDOM_STATE
from packages.regions import get_named_et_eta_regions
import os
import json
from itertools import product
import numpy as np
import pandas as pd
# from Gaugi import load as gload
# from Gaugi import save as gsave
# from kepler.pandas.readers import load as kload
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 10

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')
region_name = 'L2Calo_2017'
# dataset = 'mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2'
dataset = 'data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97'
# dataset = 'data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'
basepath = os.path.join('..', '..')
datapath = os.path.join(basepath, 'data', dataset)
filepath = os.path.join(datapath, dataset + '_et{et_region}_eta{eta}.npz')
load_func = 'gload'
crossval = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)

out_dataset = 'new_' + dataset
out_datapath =  os.path.join(basepath, 'data', out_dataset, 'data.parquet')
out_filepath =  os.path.join(out_datapath, out_dataset + '_et{et}_eta{eta}.parquet')
if not os.path.exists(out_datapath):
    os.makedirs(out_datapath)
et_eta_regions, n_ets, n_etas = get_named_et_eta_regions(region_name)

# Loading data
splitted_data = list()
for region in et_eta_regions:
    logger.info(f'Processing (et_idx, eta_idx) {(region.et_idx, region.eta_idx)}')
    if load_func == 'gload':
        data = gload(filepath.format(et=region.et_idx, eta=region.eta_idx))
        data_df = pd.DataFrame(data['data'], columns=data['features'])
        target_df = pd.DataFrame(data['target'], columns=['target'])
        data_df = pd.concat([data_df, target_df], axis=1)
    elif load_func == 'kload':
        data_df = kload(filepath.format(et=region.et_idx, eta=region.eta_idx))
    else:
        raise ValueError('Available load functions are gload and kload')
    
    spliited_data.append(data_df)
src_data_df = pd.concat(splitted_data, axis=0)
del splitted_data

# Adding ids based on original crossval
n_samples = len(src_data_df)
ifold = 0.0
for train_idx, val_idx in crossval.split(data_df.values[:,0], data_df.values[:,1]):
    logger.info(f'Adding fold {ifold}')
    new_ids = np.arange(ifold, n_samples, N_FOLDS)
    data_df.loc[val_idx, 'id'] = new_ids
    ifold += 1
src_data_df.sort_values(by='id', inplace=True)

# Saving each region
for region in et_eta_regions:
    logger.info(f'Saving (et_idx, eta_idx) {(region.et_idx, region.eta_idx)}')
    save_df = src_data_df.loc[region.get_filter(src_data_df)]
    save_df.to_parquet(out_filepath.format(et=region.et_idx, eta=region.eta_idx))