import logging
import logging.config
from packages.constants import LOGGING_CONFIG
import os
import json
from itertools import product
import numpy as np
import pandas as pd
from Gaugi import load as gload
from Gaugi import save as gsave
from kepler.pandas.readers import load as kload
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 10
RANDOM_STATE = 512

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')

old_ets = np.arange(5)
old_etas = np.arange(5)
et_key = 'L2Calo_et'
eta_key = 'L2Calo_eta'
# dataset = 'mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2'
dataset = 'data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97'
# dataset = 'data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'
basepath = os.path.join('..', '..')
datapath = os.path.join(basepath, 'data', dataset)
filepath = os.path.join(datapath, dataset + '_et{et}_eta{eta}.npz')
load_func = 'gload'
crossval = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)

out_dataset = 'new_' + dataset
out_datapath =  os.path.join(basepath, 'data', out_dataset, 'data.parquet')
out_filepath =  os.path.join(out_datapath, out_dataset + '_et{et}_eta{eta}.parquet')
if not os.path.exists(out_datapath):
    os.makedirs(out_datapath)

new_id_start = 0
for et, eta in product(old_ets, old_etas):
    logger.info(f'Processing et {et} eta {eta}')
    if load_func == 'gload':
        data = gload(filepath.format(et=et, eta=eta))
        data_df = pd.DataFrame(data['data'], columns=data['features'])
        target_df = pd.DataFrame(data['target'], columns=['target'])
        data_df = pd.concat([data_df, target_df], axis=1)
    elif load_func == 'kload':
        data_df = kload(filepath.format(et=et, eta=eta))
    else:
        raise ValueError('Available load functions are gload and kload')
    
    n_samples = len(data_df)
    ifold = 0.0
    for train_idx, val_idx in crossval.split(data_df.values[:,0], data_df.values[:,1]):
        logger.info(f'Adding fold {ifold}')
        new_ids = np.arange(ifold, n_samples, N_FOLDS)
        data_df.loc[val_idx, 'id'] = new_ids
        ifold += 1
    logger.info('Saving')
    data_df.to_parquet(out_filepath.format(et=et, eta=eta))