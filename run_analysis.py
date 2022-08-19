from kepler.pandas.menu       import ElectronSequence as Chain
from kepler.pandas.readers    import load, load_in_loop
from kepler.pandas.decorators import RingerDecorator
from typing import List
from itertools import product
import rootplotlib as rpl
import mplhep as hep
import ROOT
ROOT.gStyle.SetOptStat(0);
import numpy as np
import pandas as pd
import glob
import os
import logging
import matplotlib.pyplot as plt
plt.style.use(hep.style.ROOT)
import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser

from packages.generators import ring_percentages, RingGenerator
from packages.plotting import make_plot_fig, var2plot_func, val_label_map
from packages.utils import get_logger

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

l1seeds_per_energy = {
    24: 'L1_EM22VHI',
    26: 'L1_EM22VHI',
    60: 'L1_EM24VHI',
    140: 'L1_EM24VHI'
}

criteria_conf_names = {
    'tight': 'ElectronRingerTightTriggerConfig.conf',
    'medium': 'ElectronRingerMediumTriggerConfig.conf',
    'loose': 'ElectronRingerLooseTriggerConfig.conf',
    'vloose': 'ElectronRingerVeryLooseTriggerConfig.conf'
}

energy_chains = {
     24: 'e24_lhtight_nod0_{strategy}_ivarloose',
     26: 'e26_lhtight_nod0_{strategy}_ivarloose',
     60: 'e60_lhmedium_nod0_{strategy}_L1EM24VHI',
     140: 'e140_lhloose_nod0_{strategy}'
}

steps_choices = ['L2Calo', 'L2', 'EFCalo', 'HLT']

def parse_args():
    chain_choices = list(energy_chains.keys())
    var_choices = list(var2plot_func.keys())
    val_choices = list(val_label_map.keys())
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='datasetpath')
    parser.add_argument('--models', nargs='+', required=True, help='models directory path, can be more than one', dest='modelpaths')
    parser.add_argument('--out', required=True, help='output directory for the plots', dest='output_dir')
    parser.add_argument('--cutbased', action='store_true', help='if passed, plots the cutbased results')
    parser.add_argument('--vars', nargs='+', choices=var_choices, default=var_choices, help='x axis variables for the plots', dest='plot_vars')
    parser.add_argument('--values', nargs='+', choices=val_choices, default=val_choices, help='which values will be plotted')
    parser.add_argument('--chains', nargs='+', default=chain_choices, choices=chain_choices, help='chains to be plotted, defults to all chains', type=int, dest='chain_names')
    parser.add_argument('--steps', nargs='+', default=steps_choices, choices=steps_choices, help='trigger steps to be plotted defaults to all steps', dest='trigger_steps')
    parser.add_argument('--dev', action='store_true', help='if passed, runs the code only with the leblon region')
    parser.add_argument('--log', action='store_true', help='if passed, creates a log file with script activity')
    parser.add_argument('--markers', nargs='+', help='marker codes for each model passed', default=None, type=int)
    parser.add_argument('--colors', nargs='+', help='color codes for each model passed', default=None, type=int)
    args = parser.parse_args().__dict__
    args['chain_names'] = [energy_chains[energy] for energy in args['chain_names']]
    return args

def run_analysis(datasetpath: str, modelpaths: List[str], output_dir: str, cutbased: bool, 
         plot_vars: List[str], values: List[str], chain_names: List[str], trigger_steps: List[str], 
         dev: bool, markers: List[int], colors: List[int], **kwargs):

    analysis_logger.info('Building decorators')
    decorators = list()
    trigger_strategies = ['noringer'] if cutbased else list()
    for modelpath, criterion in product(modelpaths, criteria_conf_names.keys()):
        conf_name = criteria_conf_names[criterion]
        confpath = os.path.join(modelpath, conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        ringer_name = f'ringer_{ringer_version}'
        analysis_logger.info(f'Building decorator for {confpath}. Version: {ringer_version}')
        decorator = RingerDecorator(f'{ringer_name}_{criterion}', confpath, RingGenerator(ring_percentages[ringer_version]))
        decorators.append(decorator)
        if not ringer_name in trigger_strategies:
            trigger_strategies.append(ringer_name)
    
    analysis_logger.info('Building chains')
    chains = list()
    step_chain_names = list()
    for chain_name, strategy in product(chain_names, trigger_strategies):
        spliited_chain_name = chain_name.split('_')
        criterion = spliited_chain_name[1].replace('lh', '')
        step_chain_name = f'HLT_{chain_name.format(strategy=strategy)}'
        step_chain_names.append(step_chain_name)
        energy = int(spliited_chain_name[0][1:])
        l1seed = l1seeds_per_energy[energy]
        l2calo_column = f'{strategy}_{criterion}'
        analysis_logger.info(f'Building chain: {step_chain_name} model: {l2calo_column}')
        if strategy == 'noringer':
            chain = Chain(step_chain_name, L1Seed=l1seed)
        else:
            chain = Chain(step_chain_name, L1Seed=l1seed, l2calo_column=l2calo_column)
        chains.append(chain)

    analysis_logger.info('Loading the data')
    filename_end = '*et4_eta4.npz' if dev else '*.npz'    #If dev, loads leblon
    datafiles = glob.glob(os.path.join(datasetpath, filename_end))  
    analysis_logger.info(f'glob_path: {os.path.join(datasetpath, filename_end)}')
    data = load_in_loop(datafiles, drop_columns=drop_cols, decorators=decorators, chains=chains)

    analysis_logger.info('Making plots')
    for value, var, chain_name, step in product(values, plot_vars, chain_names, trigger_steps):
        plot_dir = os.path.join(output_dir, value, var)
        analysis_logger.info(f'Plotting value: {value}, step: {step}, chain_name: {chain_name}, var: {var}')
        make_plot_fig(data, step, chain_name, trigger_strategies, plot_dir , var, value, markers=markers, colors=colors, joblib_dump=True)

if __name__ == '__main__':
    args = parse_args()
    analysis_logger = get_logger('run_analysis', file=args['log'])
    analysis_logger.info('Script start')
    analysis_logger.info('Parsed args')
    for key, value in args.items():
        analysis_logger.info(f'{key}: {value}')
    run_analysis(**args)
