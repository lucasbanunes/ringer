from itertools import product
import mplhep as hep
import ROOT
ROOT.gStyle.SetOptStat(0);
import pandas as pd
import os
import matplotlib.pyplot as plt
from Gaugi import GeV
plt.style.use(hep.style.ROOT)
import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser
from typing import List
import joblib

from packages.plotting import var_infos, joint_plot
import packages.utils as utils
from packages.constants import CRITERIA_CONF_NAMES


def parse_args(var_choices):
    criteria_choices = list(CRITERIA_CONF_NAMES.keys())
    data_labels_choices = list(utils.LABEL_UTILITIES.keys()) + ['all']
    default_limits = [None for _ in range(len(var_choices))]
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory path', dest='datasetpath')
    parser.add_argument('--models', nargs='+', required=True, 
        help='models directory path, can be more than one', dest='modelpaths')
    parser.add_argument('--out', required=True, help='output directory for the plots', dest='output_dir')
    parser.add_argument('--vars', nargs='+', choices=var_choices, default=var_choices, 
        help='x axis variables for the plots', dest='plot_vars')
    parser.add_argument('--dlabels', nargs='+', choices=data_labels_choices, default=data_labels_choices, 
        help='If passed plots distributions of data that only attends the specified labels', dest='data_labels')
    parser.add_argument('--etcut', nargs='+', default=[None], help='upper et in GeV which there is no trigger', dest='et_cuts')
    parser.add_argument('--criteria', nargs='+', choices=criteria_choices, default=criteria_choices, 
        help='The criteria to consider on the labels')
    parser.add_argument('--var-upper', nargs='+', default=default_limits, 
        help='Upper limits to the variable axes', dest='xlims_upper')
    parser.add_argument('--var-lower', nargs='+', default=default_limits, 
        help='Lower limits to the variable axes', dest='xlims_lower')
    parser.add_argument('--model-upper', nargs='+', default=default_limits, 
        help='Upper limits to the variable axes', dest='ylims_upper')
    parser.add_argument('--model-lower', nargs='+', default=default_limits, 
        help='Lower limits to the variable axes', dest='ylims_lower')
    parser.add_argument('--dev', action='store_true', help='if passed, runs the code only with the leblon region')
    parser.add_argument('--log', action='store_true', help='if passed, creates a log file with script activity')
    parser.add_argument('--joblib', action='store_true', dest='joblib_dump',
        help='if passed, uses joblib to dump the JointGrid with the seaborn plot')
    args = parser.parse_args().__dict__
    return args

def plot_vars_distributions(datasetpath: str, modelpaths: List[str], output_dir: str, 
         plot_vars: List[str], data_labels: List[str], et_cuts: List[int], criteria:List[str],
         dev: bool, joblib_dump: bool, xlims: dict, ylims: dict, **kwargs):
    
    plot_logger.info('Getting models')
    trigger_strategies = []
    aux_conf_name = CRITERIA_CONF_NAMES['tight']
    for modelpath in modelpaths:
        confpath = os.path.join(modelpath, aux_conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", '')
        ringer_name = f'ringer_{ringer_version}'
        trigger_strategies.append(ringer_name)
    
    for trig_strat in trigger_strategies:
        parquet_file = trig_strat + '.parquet'
        chainpath = os.path.join(datasetpath, 'simulated_chains', parquet_file)
        if dev:
            chainpath = os.path.join(chainpath, f'{trig_strat}_et4_eta4.parquet')
        plot_logger.info(f'Loading: {parquet_file}')
        data = pd.read_parquet(chainpath)

        # Casting variables to GeV (They come in MeV)
        data[var_infos['pt']['col']] = data[var_infos['pt']['col']]/GeV
        data[var_infos['et']['col']] = data[var_infos['et']['col']]/GeV
        data[var_infos['et']['l2_calo_col']] = data[var_infos['et']['l2_calo_col']]/GeV

        for var, data_label, criterion, et_cut in product(plot_vars, data_labels, criteria, et_cuts):
            plot_logger.info(f'Plotting {data_label} {criterion} var {var} for {trig_strat} with et_cut {et_cut}')
            strat_name = trig_strat.capitalize().replace('_', ' ')
            text_label = var_infos[var]["label"].replace('#', '\\')
            
            if data_label == 'electron':
                title = fr'{strat_name} Output x ${text_label}$ el_lh{criterion} approved'
            elif data_label == 'jet':
                title = fr'{strat_name} Output x ${text_label}$ el_lh{criterion} reproved'
            elif data_label == 'all':
                title = fr'{strat_name} Output x ${text_label}$'
            
            try:
                et_cut = float(et_cut)
                et_cut = round(et_cut, 2)
                title += f' with $E_T < {et_cut} GeV$'
            except ValueError:
                et_cut = None
            
            jplot = joint_plot(data, y=f'{trig_strat}_{criterion}_output', x=var_infos[var]['col'], ylim=ylims[var],
                                        data_label=data_label, criterion=criterion, xlabel=fr'${text_label}$', xlim=xlims[var],
                                        ylabel=f'{trig_strat.capitalize().replace("_", " ")} Output', et_cut=et_cut,
                                        title = title, cmap="icefire_r", marg_color='black', figsize=(15,10))
            
            plot_output_dir = os.path.join(output_dir, f'output_distributions_{data_label}', var)
            plot_name = f'{var}_{trig_strat}_{data_label}_{criterion}_distribution_et_cut_{et_cut}'
            if not os.path.exists(plot_output_dir):
                os.makedirs(plot_output_dir)
            jplot.figure.savefig(os.path.join(plot_output_dir, plot_name + '.png'), transparent=False, facecolor='white')
            if joblib_dump:
                joblib.dump(jplot, os.path.join(plot_output_dir, plot_name + '.joblib'))


if __name__ == '__main__':
    var_choices = list(var_infos.keys())
    args = parse_args(var_choices)
    plot_logger = utils.get_logger('plot_vars_distributions', file=args.pop('log'))
    plot_logger.info('Script start')
    plot_logger.info('Parsed args')
    for key, value in args.items():
        plot_logger.info(f'{key}: {value}')
    xlims=dict()
    ylims=dict()
    zipped = zip(args['plot_vars'], args.pop('xlims_upper'), args.pop('xlims_lower'), args.pop('ylims_upper'), args.pop('ylims_lower'))
    for var, xupper, xlower, yupper, ylower in zipped:
        if (xupper is None) or (xlower is None):
            xlims[var]=None
        else:
            xlims[var]=(xupper, xlower)
        
        if (yupper is None) or (ylower is None):
            ylims[var]=None
        else:
            ylims[var]=(yupper, ylower)
        
    plot_vars_distributions(xlims=xlims, ylims=ylims, **args)
    plot_logger.info('Finished')
