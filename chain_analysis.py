import os
from typing import List
from itertools import product
import pandas as pd
from argparse import ArgumentParser

from ringer.root_plotting import make_plot_fig, var_infos, val_label_map, COLORS, MARKERS
from ringer.utils import get_logger
from ringer.constants import CRITERIA_CONF_NAMES, ENERGY_CHAINS, TRIG_STEPS
from ringer.data import NamedDatasetLoader


def parse_args():
    chain_choices = list(ENERGY_CHAINS.keys())
    var_choices = list(var_infos.keys())
    val_choices = list(val_label_map.keys())
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='dataset_name')
    parser.add_argument('--strategies', nargs='+', required=True, help='distinct trigger strategies to analyze', dest='trigger_strategies')
    parser.add_argument('--out', required=True, help='output directory for the plots', dest='output_dir')
    parser.add_argument('--vars', nargs='+', choices=var_choices, default=var_choices, help='x axis variables for the plots', dest='vars2plot')
    parser.add_argument('--values', nargs='+', choices=val_choices, default=val_choices, help='which values will be plotted', dest='values2plot')
    parser.add_argument('--chains', nargs='+', default=chain_choices, choices=chain_choices, help='chains to be plotted, defaults to all chains', type=int, dest='chain_names')
    parser.add_argument('--steps', nargs='+', default=TRIG_STEPS, choices=TRIG_STEPS, help='trigger steps to be plotted defaults to all steps', dest='trigger_steps')
    parser.add_argument('--dev', action='store_true', help='if passed, runs the code only with the leblon region')
    parser.add_argument('--log', action='store_true', help='if passed, creates a log file with script activity record')
    parser.add_argument('--markers', nargs='+', help='marker codes for each model passed', default=None, type=int)
    parser.add_argument('--colors', nargs='+', help='color codes for each model passed', default=None, type=int)
    args = parser.parse_args().__dict__
    args['chain_names'] = [ENERGY_CHAINS[energy] for energy in args['chain_names']]
    return args

def get_data_df_load_cols(vars2plot: List[str]) -> List[str]:
    load_cols = ['id', 'target'] + [f'el_lh{criterion}' for criterion in CRITERIA_CONF_NAMES.keys()]
    for var in vars2plot:
        load_cols.append(var_infos[var]['col'])
        try:
            load_cols.append(var_infos[var]['l2_calo_col'])
        except KeyError:
            pass
    return load_cols

def get_plot_df(dataset_name: str, load_cols: List[str], trigger_strategies: List[str], dev: bool) -> pd.DataFrame:
    dataset_loader = NamedDatasetLoader(dataset_name, dev)
    analysis_logger.info("Loading data_df")
    plot_df = dataset_loader.load_data_df(load_cols)
    for trig_strat in trigger_strategies:
        analysis_logger.info(f"Loading strategy_df {trig_strat}")
        strategy_df = dataset_loader.load_strategy_df(trig_strat)
        plot_df = pd.merge(plot_df, strategy_df, on='id', how='inner')
    return plot_df

def make_all_plot_figs(plot_df: pd.DataFrame, trigger_strategies: List[str],
                       values2plot: List[str], vars2plot: List[str],
                       chain_names: List[str], trigger_steps: List[str],
                       colors: List[str], markers: List[str],
                       output_dir: str):
    n_strats = len(trigger_strategies)
    colors = COLORS[:n_strats] if colors is None else colors
    markers = MARKERS[:n_strats] if markers is None else markers
    analysis_logger.info('Making plots')
    for value, var, chain_name, step in product(values2plot, vars2plot, chain_names, trigger_steps):
        plot_dir = os.path.join(output_dir, value, var)
        analysis_logger.info(f'Plotting value: {value}, step: {step}, chain_name: {chain_name}, var: {var}')
        make_plot_fig(plot_df, step, chain_name, trigger_strategies, 
                    plot_dir , var, value, joblib_dump=True,
                    markers=markers, colors=colors)

def plot_effs(dataset_name: str, output_dir: str, trigger_strategies: List[str],
              vars2plot: List[str], values2plot: List[str], chain_names: List[str], trigger_steps: List[str], 
              dev: bool, markers: List[int], colors: List[int], **kwargs):
    
    load_cols = get_data_df_load_cols(vars2plot)
    plot_df = get_plot_df(dataset_name, load_cols, trigger_strategies, dev)
    make_all_plot_figs(plot_df, trigger_strategies,
                       values2plot, vars2plot,
                       chain_names, trigger_steps,
                       colors, markers, output_dir)

if __name__ == '__main__':
    args = parse_args()
    analysis_logger = get_logger('plot_effs', file=args['log'])
    analysis_logger.info('Script start')
    analysis_logger.info('Parsed args')
    for key, value in args.items():
        analysis_logger.info(f'{key}: {value}')
    plot_effs(**args)
    analysis_logger.info('Finished')
