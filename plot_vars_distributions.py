import seaborn as sns
from itertools import product
import rootplotlib as rpl
import mplhep as hep
import root_numpy
import ROOT
ROOT.gStyle.SetOptStat(0);
import numpy as np
import pandas as pd
import glob
import os
import logging
import matplotlib.pyplot as plt
from matplotlib import gridspec
from Gaugi import GeV
plt.style.use(hep.style.ROOT)
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

from packages.generators import ring_percentages, RingGenerator
from packages.plotting import make_plot_fig, cached_root2fig, root2fig, var_infos, plot_distribution
import packages.utils as utils
from packages.constants import DROP_COLS, L1SEEDS_PER_ENERGY, CRITERIA_CONF_NAMES, ENERGY_CHAINS, TRIG_STEPS, HOME_PATH


def parse_args():
    chain_choices = list(ENERGY_CHAINS.keys())
    var_choices = list(var_infos.keys())
    val_choices = list(val_label_map.keys())
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='datasetpath')
    parser.add_argument('--models', nargs='+', required=True, help='models directory path, can be more than one', dest='modelpaths')
    parser.add_argument('--out', required=True, help='output directory for the plots', dest='output_dir')
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

def plot_vars_distributions(datasetpath: str, modelpaths: List[str], output_dir: str, 
         plot_vars: List[str], data_labels: List[str],
         dev: bool, **kwargs):

    plot_logger.info('Getting models')
    trigger_strategies = []
    aux_conf_name = CRITERIA_CONF_NAMES['tight']
    for modelpath in modelpaths:
        confpath = os.path.join(modelpath, aux_conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        ringer_name = f'ringer_{ringer_version}'
        trigger_strategies.append(ringer_name)
    
    plots = dict()
    lims = dict(pt=(0,2000), et=(0,300), dr=(0,0.6), mu=(0,60), eta=(-2.5,2.5))
    for trig_strat in trigger_strategies:
        var_plots = dict()
        plots[trig_strat] = var_plots
        parquet_file = trig_strat + '.parquet'
        chainpath = os.path.join(datasetpath, 'simulated_chains', parquet_file)
        if dev:
            chainpath = os.path.join(chainpath, f'{trig_strat}_et4_eta4.parquet')
        plot_logger.info(f'Loading: {parquet_file}')
        data = pd.read_parquet(chainpath)
        data[var_infos['pt']['col']] = data[var_infos['pt']['col']]/GeV
        data[var_infos['et']['col']] = data[var_infos['et']['col']]/GeV
        for var, data_label in product(plot_vars, data_labels):
            plot_logger.info(f'Plotting var {var} for {trig_strat}')
            strat_name = trig_strat.capitalize().replace('_', ' ')
            text_label = var_infos[var]["label"].replace('#', '\\')
            fig, axes = plot_distribution(data, f'{trig_strat}_tight_output', var_infos[var]['col'], 
                                        xlim=lims[var], xlabel=fr'${text_label}$', ylabel='Output', et_cut=300, data_label=data_label,
                                        title = fr'{strat_name} Output x ${text_label}$ for $E_T < 300 GeV$', cmap="icefire_r", hist1d_color='black')
            fig.tight_layout()
            plot_output_dir = os.path.join(output_dir, f'output_distributions_{data_label}', var)
            if not os.path.exists(plot_output_dir):
                os.makedirs(plot_output_dir)
            fig.savefig(os.path.join(plot_output_dir, f'{var}_{trig_strat}_{data_label}_distribution.png'), transparent=False, facecolor='white')
            var_plots[var] = (fig, axes)

if __name__ == '__main__':
    args = parse_args()
    plot_logger = get_logger('plot_vars_distributions', file=log)
    plot_logger.info('Script start')
    plot_logger.info('Parsed args')
    for key, value in args.items():
        plot_logger.info(f'{key}: {value}')
    run_analysis(**args)
    plot_logger.info('Finished')
