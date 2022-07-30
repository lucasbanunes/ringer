from kepler.pandas.menu       import ElectronSequence as Chain
from kepler.pandas.readers    import load, load_in_loop
from kepler.pandas.decorators import create_ringer_v8_decorators, create_ringer_v9_decorators, RingerDecorator
from kepler.pandas.decorators import create_ringer_v8_new_decorators, create_ringer_v8_half_fast_decorators, create_ringer_v8_34_decorators, create_ringer_v8_half_decorators
from kepler.pandas.decorators import create_ringer_v20_decorators

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


import matplotlib.pyplot as plt
from matplotlib import gridspec
%matplotlib inline

import mplhep as hep

import warnings
warnings.filterwarnings('ignore')
plt.style.use(hep.style.ROOT)

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

boosted_chains = [
    "HLT_e24_lhtight_nod0_{strategy}_v20_ivarloose",
    "HLT_e26_lhtight_nod0_{strategy}_ivarloose",,
    "HLT_e60_lhmedium_nod0_{strategy}_L1EM24VHI",
    "HLT_e140_lhloose_nod0_{strategy}"
]

l1seeds = [
    'L1_EM22VHI',
    'L1_EM22VHI',
    'L1_EM24VHI',
    'L1_EM24VHI'
]

conf_names = [
    'ElectronRingerLooseTriggerConfig.conf',
    'ElectronRingerMediumTriggerConfig.conf',
    'ElectronRingerTightTriggerConfig.conf',
    'ElectronRingerVeryLooseTriggerConfig.conf'
]

def main(datasetpath, modelpaths):
    datafiles = glob.glob(os.path.join(datasetpath, '*.npz'))

    decorators = list()
    strategies = list()
    last_version = ''
    for modelpath, conf_name in modelpaths, conf_names:
        confpath = os.path.join(modelpath, conf_name)
        env = ROOT.TEnv(confpath)
        version = env.GetValue("__version__", '')
        if last_version != version:
            strategies.append(f'ringer_{version}')
            last_version=version

    # Load the data
    #data_df = load_in_loop(datafiles, drop_columns=drop_columns, decorators=decorators, chains=chains)

