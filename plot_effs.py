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

from packages.plotting import make_plot_fig, var_infos, val_label_map, COLORS, MARKERS
from packages.utils import get_logger
from packages.constants import CRITERIA_CONF_NAMES, ENERGY_CHAINS, TRIG_STEPS


def parse_args():
    chain_choices = list(ENERGY_CHAINS.keys())
    var_choices = list(var_infos.keys())
    val_choices = list(val_label_map.keys())
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='datasetpath')
    parser.add_argument('--models', nargs='+', required=True, help='models directory path, can be more than one', dest='modelpaths')
    parser.add_argument('--out', required=True, help='output directory for the plots', dest='output_dir')
    parser.add_argument('--cutbased', action='store_true', help='if passed, plots the cutbased results')
    parser.add_argument('--vars', nargs='+', choices=var_choices, default=var_choices, help='x axis variables for the plots', dest='plot_vars')
    parser.add_argument('--values', nargs='+', choices=val_choices, default=val_choices, help='which values will be plotted')
    parser.add_argument('--chains', nargs='+', default=chain_choices, choices=chain_choices, help='chains to be plotted, defults to all chains', type=int, dest='chain_names')
    parser.add_argument('--steps', nargs='+', default=TRIG_STEPS, choices=TRIG_STEPS, help='trigger steps to be plotted defaults to all steps', dest='trigger_steps')
    parser.add_argument('--dev', action='store_true', help='if passed, runs the code only with the leblon region')
    parser.add_argument('--log', action='store_true', help='if passed, creates a log file with script activity')
    parser.add_argument('--markers', nargs='+', help='marker codes for each model passed', default=None, type=int)
    parser.add_argument('--colors', nargs='+', help='color codes for each model passed', default=None, type=int)
    args = parser.parse_args().__dict__
    args['chain_names'] = [ENERGY_CHAINS[energy] for energy in args['chain_names']]
    return args

def plot_effs(datasetpath: str, modelpaths: List[str], output_dir: str, cutbased: bool, 
         plot_vars: List[str], values: List[str], chain_names: List[str], trigger_steps: List[str], 
         dev: bool, markers: List[int], colors: List[int], **kwargs):

    trigger_strategies = ['noringer'] if cutbased else list()
    aux_conf_name = CRITERIA_CONF_NAMES['tight']
    for modelpath in modelpaths:
        confpath = os.path.join(modelpath, aux_conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        ringer_name = f'ringer_{ringer_version}'
        trigger_strategies.append(ringer_name)
    
    analysis_logger.info('Loading chains')
    dataset_dir, dataset_name = os.path.split(datasetpath)
    dataset_name = dataset_name.replace('.parquet', '')
    load_cols = ['id', 'target'] + [f'el_lh{criterion}' for criterion in CRITERIA_CONF_NAMES.keys()]
    for var in plot_vars:
        load_cols.append(var_infos[var]['col'])
        try:
            load_cols.append(var_infos[var]['l2_calo_col'])
        except KeyError:
            pass
    plot_data = pd.read_parquet(datasetpath, columns=load_cols)
    for trig_strat in trigger_strategies:
        parquet_file = trig_strat + '.parquet'
        chainpath = os.path.join(dataset_dir, 'sim_chains_'+dataset_name, parquet_file)
        if dev:
            chainpath = os.path.join(chainpath, f'{trig_strat}_et4_eta4.parquet')
        analysis_logger.info(f'Loading: {chainpath}')
        plot_data = plot_data.join(pd.read_parquet(chainpath), on='id', how='outer', rsuffix=f'_{trig_strat}')
    plot_data.drop(plot_data.columns[plot_data.columns.str.startswith('id_')], axis=1, inplace=True)
    
    n_strats = len(trigger_strategies)
    colors = COLORS[:n_strats] if colors is None else colors
    markers = MARKERS[:n_strats] if markers is None else markers
    analysis_logger.info('Making plots')
    for value, var, chain_name, step in product(values, plot_vars, chain_names, trigger_steps):
        plot_dir = os.path.join(output_dir, value, var)
        analysis_logger.info(f'Plotting value: {value}, step: {step}, chain_name: {chain_name}, var: {var}')
        make_plot_fig(plot_data, step, chain_name, trigger_strategies, 
                    plot_dir , var, value, joblib_dump=True,
                    markers=markers, colors=colors)

if __name__ == '__main__':
    args = parse_args()
    analysis_logger = get_logger('plot_effs', file=args['log'])
    analysis_logger.info('Script start')
    analysis_logger.info('Parsed args')
    for key, value in args.items():
        analysis_logger.info(f'{key}: {value}')
    plot_effs(**args)
    analysis_logger.info('Finished')
