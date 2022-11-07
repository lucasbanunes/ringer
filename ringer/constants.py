import os
import numpy as np
from collections import OrderedDict

VAR_INFOS_DTYPES = dict(
    name='str', label='str', tyoe='category', lower_lim='float', upper_lim='float', 
    l2calo='str', offline='str', TaP='str', description='str'
)

DROP_COLS = [
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

L1SEEDS_PER_ENERGY = {
    24: 'L1_EM22VHI',
    26: 'L1_EM22VHI',
    60: 'L1_EM24VHI',
    140: 'L1_EM24VHI'
}

CRITERIA_CONF_NAMES = {
    'tight': 'ElectronRingerTightTriggerConfig.conf',
    'medium': 'ElectronRingerMediumTriggerConfig.conf',
    'loose': 'ElectronRingerLooseTriggerConfig.conf',
    'vloose': 'ElectronRingerVeryLooseTriggerConfig.conf'
}

ENERGY_CHAINS = {
     24: 'e24_lhtight_nod0_{strategy}_ivarloose',
     26: 'e26_lhtight_nod0_{strategy}_ivarloose',
     60: 'e60_lhmedium_nod0_{strategy}_L1EM24VHI',
     140: 'e140_lhloose_nod0_{strategy}'
}

TRIG_STEPS = ['L2Calo', 'L2', 'EFCalo', 'HLT']
HOME_PATH = os.path.expanduser('~')

STEP_PREFIX = {
    'L2Calo': 'trig_L2_cl_'
}

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'streamFormat': {
            'format': '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        },
        'csvformat': {
            'format': '%(asctime)s;%(process)d;%(levelname)s;%(module)s;%(funcName)s;%(message)s'
        }
    },
    'handlers': {
        'stream': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'streamFormat',
        }
    },
    'loggers': {
        'ringer_debug': {
            'level': 'DEBUG',
            'handlers': ['stream'],
            'propagate': True
        }
    }
}

NAMED_ET_ETA_BINS = {
    'L2Calo_2017': {
        'et_bins': [15,20,30,40,50,np.inf], 
        'eta_bins': [0,0.8,1.37,1.54,2.37,2.5],
        'et_inclusives': ['left','left','left','left','both'],
        'eta_inclusives': ['left','left','left','left','both'],
        'et_key': 'trig_L2_cl_et',
        'eta_key': 'trig_L2_cl_eta'
    },
    'L2Calo_2017_alt': {
    'et_bins': [15,20,30,40,50,np.inf], 
    'eta_bins': [0,0.8,1.37,1.54,2.37,2.5],
    'et_inclusives': ['left','left','left','left','both'],
    'eta_inclusives': ['left','left','left','left','both'],
    'et_key': 'L2Calo_et',
    'eta_key': 'L2Calo_eta'
    }
}

RANDOM_STATE = 512

RING_COL_NAME = 'trig_L2_cl_ring_{ring_num}'
RINGS_LAYERS = OrderedDict(
    presample = (0,8),
    em1 = (8,72),
    em2 = (72, 80),
    em3 = (80,88),
    had1 = (88,92),
    had2 = (92,96),
    had3 = (96,100),
)

NAMED_DATASETS = {
    'MC16 Boosted': os.path.join('..', '..', 'data', 'new_mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2'),
    '2017 Medium': os.path.join('..', '..', 'data', 'data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97'),
    '2017 VLoose': os.path.join('..', '..', 'data', 'data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins')
}