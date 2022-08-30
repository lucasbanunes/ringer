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
from collections import defaultdict
from copy import deepcopy
from argparse import ArgumentParser
import json
from Gaugi import GeV

from packages.generators import ring_percentages, RingGenerator
from packages.plotting import make_plot_fig, var_infos, val_label_map
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

trigger_steps = ['L2Calo', 'L2', 'EFCalo', 'HLT']

et_bins = [15, 20, 30, 40, 50, 1000000]
et_bins_idxs = range(len(et_bins)-1)
eta_bins = [0.0, 0.8, 1.37, 1.54, 2.37, 2.50]
eta_bins_idxs = range(len(eta_bins)-1)

def parse_args():
    chain_choices = list(energy_chains.keys())
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='datasetpath')
    parser.add_argument('--models', nargs='+', required=True, help='models directory path, can be more than one', dest='modelpaths')
    parser.add_argument('--cutbased', action='store_true', help='if passed, plots the cutbased results')
    parser.add_argument('--chains', nargs='+', default=chain_choices, choices=chain_choices, help='chains to be plotted, defults to all chains', type=int, dest='chain_names')
    parser.add_argument('--dev', action='store_true', help='if passed, runs the code only with the leblon region')
    parser.add_argument('--log', action='store_true', help='if passed, creates a log file with script activity')
    args = parser.parse_args().__dict__
    args['chain_names'] = [energy_chains[energy] for energy in args['chain_names']]
    return args

def simulate(datasetpath: str, modelpaths: List[str], cutbased: bool, 
        chain_names: List[str], dev: bool, **kwargs):

    simulation_logger.info('Building decorators')
    decorators = list()
    trigger_strategies = ['noringer'] if cutbased else list()
    strategy_cols = defaultdict(list)
    for modelpath, criterion in product(modelpaths, criteria_conf_names.keys()):
        conf_name = criteria_conf_names[criterion]
        confpath = os.path.join(modelpath, conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        ringer_name = f'ringer_{ringer_version}'
        strat_criterion = f'{ringer_name}_{criterion}'
        simulation_logger.info(f'Building decorator for {confpath}. Version: {ringer_version}')
        decorator = RingerDecorator(strat_criterion, confpath, RingGenerator(ring_percentages[ringer_version]))
        decorators.append(decorator)
        strategy_cols[ringer_name].append(strat_criterion)
        strategy_cols[ringer_name].append(strat_criterion + '_output')
        if not ringer_name in trigger_strategies:
            trigger_strategies.append(ringer_name)
    
    simulation_logger.info('Building chains')
    chains = list()
    step_chain_names = list()
    for chain_name, strategy in product(chain_names, trigger_strategies):
        spliited_chain_name = chain_name.split('_')
        criterion = spliited_chain_name[1].replace('lh', '')
        step_chain_name = f'HLT_{chain_name.format(strategy=strategy)}'
        step_chain_names.append(step_chain_name)
        strategy_cols[strategy].append(step_chain_name)
        for trigger_step in trigger_steps:
            if trigger_step != 'HLT':
                lower_chain = step_chain_name.replace('HLT', trigger_step)
                step_chain_names.append(lower_chain)
                strategy_cols[strategy].append(lower_chain)
        energy = int(spliited_chain_name[0][1:])
        l1seed = l1seeds_per_energy[energy]
        l2calo_column = f'{strategy}_{criterion}'
        simulation_logger.info(f'Building chain: {step_chain_name} model: {l2calo_column}')
        if strategy == 'noringer':
            chain = Chain(step_chain_name, L1Seed=l1seed)
        else:
            chain = Chain(step_chain_name, L1Seed=l1seed, l2calo_column=l2calo_column)
        chains.append(chain)

    simulation_logger.info('Simulating')
    filename_end = '*et4_eta4.npz' if dev else '*.npz'    #If dev, load only a part of the dataset
    datafiles = glob.glob(os.path.join(datasetpath, filename_end))  
    simulation_logger.info(f'glob_path: {os.path.join(datasetpath, filename_end)}')
    data = load_in_loop(datafiles, drop_columns=drop_cols, decorators=decorators, chains=chains)

    simulation_logger.info('Saving_data')
    output_dir = os.path.join(datasetpath, 'simulated_chains')
    save_cols = [f'el_lh{criterion}' for criterion in criteria_conf_names.keys()]
    save_cols += ['trig_L2_el_pt', 'avgmu', 'el_et', 'el_eta', 'el_TaP_deltaR']
    save_cols += ['target']
    last_strat = None
    for strategy, et_bin_idx, eta_bin_idx in product(strategy_cols.keys(), et_bins_idxs, eta_bins_idxs):
        outname = f'{strategy}_et{et_bin_idx}_eta{eta_bin_idx}'
        simulation_logger.info(f'Generating {outname}')
        strategy_out = os.path.join(output_dir, strategy + '.parquet')
        if not os.path.exists(strategy_out):
            os.makedirs(strategy_out)

        selection_cols = strategy_cols[strategy] + save_cols
        if last_strat != strategy:
            with open(os.path.join(output_dir, f'{strategy}_cols.json'), 'w') as json_file:
                json.dump(selection_cols, json_file, indent=4)
            last_strat = deepcopy(strategy)

        et_min, et_max = et_bins[et_bin_idx:et_bin_idx+2]
        eta_min, eta_max = eta_bins[eta_bin_idx:eta_bin_idx+2]
        bin_selector = (data['el_et'] >= (et_min*GeV)) & (data['el_et'] < (et_max*GeV))
        bin_selector = bin_selector & (data['el_eta'].abs() >= eta_min) & (data['el_eta'].abs() < eta_max)
        selected_data = data.loc[bin_selector, selection_cols]

        simulation_logger.info(f'Saving: {outname}')
        selected_data.to_parquet(os.path.join(strategy_out, outname + '.parquet'))
    
    return data


if __name__ == '__main__':
    args = parse_args()
    simulation_logger = get_logger('simulate_chains', file=args['log'])
    simulation_logger.info('Script start')
    simulation_logger.info('Parsed args')
    for key, value in args.items():
        simulation_logger.info(f'{key}: {value}')
    simulate(**args)
    simulation_logger.info('Finished')
