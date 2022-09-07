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
from packages.constants import DROP_COLS, L1SEEDS_PER_ENERGY, CRITERIA_CONF_NAMES, ENERGY_CHAINS, TRIG_STEPS

et_bins = [15, 20, 30, 40, 50, 1000000]
eta_bins = [0.0, 0.8, 1.37, 1.54, 2.37, 2.50]

def parse_args():
    chain_choices = list(ENERGY_CHAINS.keys())
    et_choices = list(range(len(et_bins)-1))
    eta_choices = list(range(len(eta_bins)-1))
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='datasetpath')
    parser.add_argument('--models', nargs='+', required=True, help='models directory path, can be more than one', dest='modelpaths')
    parser.add_argument('--cutbased', action='store_true', help='if passed, plots the cutbased results')
    parser.add_argument('--chains', nargs='+', default=chain_choices, choices=chain_choices, help='chains to be plotted, defults to all chains', type=int, dest='chain_names')
    parser.add_argument('--dev', action='store_true', help='if passed, runs the code only with the leblon region')
    parser.add_argument('--log', action='store_true', help='if passed, creates a log file with script activity')
    parser.add_argument('--ets', nargs='+', choices=et_choices, default=et_choices, type=int,
        help='et regions to simulate')
    parser.add_argument('--etas', nargs='+', choices=eta_choices, default=eta_choices, type=int,
        help='eta regions to simulate')
    args = parser.parse_args().__dict__
    args['chain_names'] = [ENERGY_CHAINS[energy] for energy in args['chain_names']]
    return args

def simulate(datasetpath: str, modelpaths: List[str], cutbased: bool, 
        chain_names: List[str], dev: bool, ets: List[int], etas: List[int], **kwargs):

    simulation_logger.info('Building decorators')
    decorators = list()
    trigger_strategies = ['noringer'] if cutbased else list()
    strategy_cols = defaultdict(list)
    for modelpath, criterion in product(modelpaths, CRITERIA_CONF_NAMES.keys()):
        conf_name = CRITERIA_CONF_NAMES[criterion]
        confpath = os.path.join(modelpath, conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        if ringer_version == 'should_be_filled':
            raise ValueError(f'The model from {modelpath} does not have a version. Please fill it. Version found: {ringer_version}')
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
        for trigger_step in TRIG_STEPS:
            if trigger_step != 'HLT':
                lower_chain = step_chain_name.replace('HLT', trigger_step)
                step_chain_names.append(lower_chain)
                strategy_cols[strategy].append(lower_chain)
        energy = int(spliited_chain_name[0][1:])
        l1seed = L1SEEDS_PER_ENERGY[energy]
        l2calo_column = f'{strategy}_{criterion}'
        simulation_logger.info(f'Building chain: {step_chain_name} model: {l2calo_column}')
        if strategy == 'noringer':
            chain = Chain(step_chain_name, L1Seed=l1seed)
        else:
            chain = Chain(step_chain_name, L1Seed=l1seed, l2calo_column=l2calo_column)
        chains.append(chain)

    filename_end = '_et{et}_eta{eta}.parquet'
    dataset_dir, datasetname = os.path.split(datasetpath)
    dataset_name = datasetname.replace('.parquet', '')
    simulation_logger.info('Reading schema')
    with open(os.path.join(dataset_dir, dataset_name + '_schema.json'), 'r') as json_file:
        data_cols = list(json.load(json_file).keys())
    load_cols = [col for col in data_cols if col not in DROP_COLS]
    output_dir = os.path.join(dataset_dir, 'sim_chains_' + dataset_name)
    last_strat = None
    # If dev, load only a part of the dataset for last_bin understanding see the if clause 
    # that uses it bellow
    ibins = product([4], [4]) if dev else product(ets, etas)
    last_bin = 0 if dev else len(ets)*len(etas) - 1
    for i, bins in enumerate(ibins):
        et_bin_idx, eta_bin_idx = bins
        start_msg = f'et {et_bin_idx} eta {eta_bin_idx} '
        _, datasetname = os.path.split(datasetpath)
        filepath = os.path.join(datasetpath, datasetname + filename_end.format(et=et_bin_idx, eta=eta_bin_idx))
        simulation_logger.info(start_msg + f'loading {filepath}')
        data = pd.read_parquet(filepath, columns=load_cols)
        simulation_logger.info(start_msg + 'simulating')
        for decorator in decorators:
            decorator.apply(data)
        for chain in chains:
            chain.apply(data)
        
        for strategy in strategy_cols.keys():
            outname = f'{strategy}_et{et_bin_idx}_eta{eta_bin_idx}'
            simulation_logger.info(start_msg + f'generating {outname}')
            strategy_out = os.path.join(output_dir, strategy + '.parquet')
            if not os.path.exists(strategy_out):
                os.makedirs(strategy_out)

            selection_cols = strategy_cols[strategy] + ['id']   # Saves the id for future joining if necessary
            selected_data = data[selection_cols]
            # This ensures that the schema is saved only at the end
            if last_strat != strategy and i == last_bin:
                with open(os.path.join(output_dir, f'{strategy}_schema.json'), 'w') as json_file:
                    json.dump(selected_data.dtypes.astype(str).to_dict(), json_file, indent=4)
                last_strat = deepcopy(strategy)

            simulation_logger.info(start_msg + f'saving: {outname}')
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
