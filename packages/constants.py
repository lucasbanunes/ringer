import os
import numpy as np
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