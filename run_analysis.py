from kepler.pandas.menu       import ElectronSequence as Chain
from kepler.pandas.readers    import load, load_in_loop
from kepler.pandas.decorators import RingerDecorator
from collections import defaultdict
from typing import List

import kepler
from itertools import product
import tqdm
import rootplotlib as rpl
import mplhep as hep
import root_numpy
import ROOT
ROOT.gStyle.SetOptStat(0);
import array

import numpy as np
import pandas as pd
import collections
import glob
import os
from pprint import pprint
from copy import deepcopy
import gc
import logging
from datetime import datetime


import matplotlib.pyplot as plt
from matplotlib import gridspec
%matplotlib inline

import mplhep as hep

import warnings
warnings.filterwarnings('ignore')
plt.style.use(hep.style.ROOT)

from packages.generators import version_generators
from packages.plotting import make_plot_fig

drop_cols = drop_columns = [
                    'RunNumber', 
                    'trig_L2_cl_e2tsts1',
                    'trig_L2_el_hastrack',
                    'trig_L2_el_eta',
                    'trig_L2_el_phi',
                    'trig_L2_el_caloEta',
                    'trig_L2_el_trkClusDeta',
                    'trig_L2_el_trkClusDphi',
                    'trig_L2_el_etOverPt',
                    'trig_EF_cl_hascluster',
                    'trig_EF_cl_eta',
                    'trig_EF_cl_etaBE2',
                    'trig_EF_cl_phi',     
                    'trig_EF_el_hascand',
                    'trig_EF_el_eta',
                    'trig_EF_el_etaBE2',
                    'trig_EF_el_phi',
                    'trig_EF_el_rhad1',
                    'trig_EF_el_rhad',
                    'trig_EF_el_f3',
                    'trig_EF_el_weta2',
                    'trig_EF_el_rphi',
                    'trig_EF_el_reta',
                    'trig_EF_el_wtots1',
                    'trig_EF_el_eratio',
                    'trig_EF_el_f1',
                    'trig_EF_el_hastrack',
                    'trig_EF_el_deltaEta1',
                    'trig_EF_el_deltaPhi2',
                    'trig_EF_el_deltaPhi2Rescaled',
                    'el_etaBE2',
                    'el_numberOfBLayerHits',
                    'el_numberOfPixelHits',
                    'el_numberOfTRTHits',
                    'el_trans_TRT_PID',
                    'el_deltaPhi2',
                    'el_TaP_Mass',
                ]

# base_chain_names = ['e24_lhtight_nod0_{RINGER}_ivarloose',
#               'e26_lhtight_nod0_{RINGER}_ivarloose',
#               'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
#               'e140_lhloose_nod0_{RINGER}'
# ]

# trigger_steps = ['L2Calo', 'L2', 'EFCalo', 'HLT']

# boosted_chains = [
#     "HLT_e24_lhtight_nod0_{strategy}_v20_ivarloose",
#     "HLT_e26_lhtight_nod0_{strategy}_ivarloose",,
#     "HLT_e60_lhmedium_nod0_{strategy}_L1EM24VHI",
#     "HLT_e140_lhloose_nod0_{strategy}"
# ]

l1seeds_per_energy = {
    24: 'L1_EM22VHI',
    26: 'L1_EM22VHI',
    60: 'L1_EM24VHI',
    140: 'L1_EM24VHI'
}

conf_names = [
    'ElectronRingerLooseTriggerConfig.conf',
    'ElectronRingerMediumTriggerConfig.conf',
    'ElectronRingerTightTriggerConfig.conf',
    'ElectronRingerVeryLooseTriggerConfig.conf'
]

criteria = ['tight', 'medium', 'loose', 'vloose']

# plot_vars = ['et', 'eta', 'pt', 'mu']

def get_logger():
    logger = logging.getLogger('analysis_logger')
    logger.setLevel(logging.INFO)
    now = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = f'run_analysis_{now}'
    file_handler = logging.FileHandler(log_filename, mode='w')
    formatter = logging.Formatter('%(asctime)s,%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def main(datasetpath: str, modelpaths: List[str], output_dir: str, 
         plot_vars: List[str], chain_names: List[str], trigger_steps: List[str], 
         dev=False):
    """
    datasetpath: str
    """
    # Getting logger
    analysis_logger = get_logger()

    analysis_logger.info('Building decorators')
    decorators = list()
    trigger_strategies = list()
    for modelpath, conf_criterion in product(modelpaths, zip(conf_names, criteria)):
        conf_name, criterion = conf_criterion
        confpath = os.path.join(modelpath, conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        ringer_name = f'ringer_{ringer_version}'
        decorator = RingerDecorator(f'{ringer_name}_{criterion}', confpath, version_generators[ringer_version])
        decorators.append(decorator)
        if not ringer_version in trigger_strategies:
            trigger_strategies.append(ringer_name)
    
    analysis_logger.info('Building chains')
    chains = list()
    step_chain_names = list()
    for step, chain_name, strategy, criterion in product(trigger_steps, chain_names, trigger_strategies, criteria):
        step_chain_name = f'{step}_{chain_name.format(strategy)}'
        step_chain_names.append(step_chain_name)
        energy = int(chain_name.split('_')[0][1:])
        l1seed = l1seeds_per_energy[energy]
        l2calo_column = f'{strategy}_{criterion}'
        chain = Chain(step_chain_name, L1Seed=l1seed, l2calo_column=l2calo_column)
        chains.append(chain)

    analysis_logger.info('Loading the data')
    datafiles = glob.glob(os.path.join(datasetpath, '*.npz'))
    data = load_in_loop(datafiles, drop_columns=drop_cols, decorators=decorators, chains=chains)

    analysis_logger.info('Making plots')
    figs = dict()
    for fake, step, chain_name, var in tqdm(product([True, False], trigger_steps, chain_names, plot_vars)):
        val_name = 'fr' if fake else 'pd'
        plot_dir = os.path.join(output_dir, val_name, var)
        plot_name, fig = make_plot_fig(data, step, chain_name, trigger_strategies, plot_dir , var, fake)
        analysis_logger.info(f'Plotted {plot_name}')
        figs[plot_name] = fig
