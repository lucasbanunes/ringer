from Gaugi.constants import GeV, MeV
import rootplotlib as rpl

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import root_numpy
import ROOT
ROOT.gStyle.SetOptStat(0);
import array
import os
from itertools import product
import joblib
import packages.utils as utils

# Root colors
# More info at https://root.cern.ch/doc/master/classTColor.html
COLORS = [
    ROOT.kBlack, 
    ROOT.kBlue+1, 
    ROOT.kGreen+1, 
    ROOT.kRed+1,  
    ROOT.kMagenta+1,
    ROOT.kYellow+1,
    ROOT.kCyan+2
]

# Root markers
# More info at https://root.cern.ch/doc/master/classTAttMarker.html
MARKERS = [
    ROOT.kFullDiamond, 
    ROOT.kFullTriangleUp, 
    ROOT.kFullTriangleDown, 
    ROOT.kOpenStar, 
    ROOT.kFullCircle, 
    ROOT.kFullSquare,
    ROOT.kFullCross
]

def hist1d( name, values, bins, density=False ):
    counts, dummy = np.histogram(values, bins=bins, density=density )
    hist = ROOT.TH1F( name, '', len(bins)-1, array.array('d',bins))
    root_numpy.array2hist(counts, hist)
    return hist

def add_legend(x, y, legends):
    rpl.add_legend( legends, x, y, x+0.98, y+0.20, textsize=12, option='p' )

def make_et_plot(dataframe, chain, chain_step, l2suffix, value):
    from Gaugi.constants import GeV

    m_bins = [4,7,10,15,20,25,30,35,40,45,50,60,80,150,300] # et_bins
    et_cut  = int(chain.split('_')[1][1:])
    offline = chain.split('_')[2]
    if value == 'fr':
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    elif value == 'pd':
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    else:
        raise ValueError(f'There is no handler for value {value}')
    
    step_decision = chain
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('et_num', passed['el_et']/GeV, m_bins )
    h_den = hist1d('et_den', total['el_et']/GeV, m_bins )
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)

def make_eta_plot(dataframe, chain, chain_step, l2suffix, value):

    m_bins = [-2.47,-2.37,-2.01,-1.81,-1.52,-1.37,-1.15,-0.80,-0.60,-0.10,0.00,
              0.10, 0.60, 0.80, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
    
    et_cut  = int(chain.split('_')[1][1:])
    offline = chain.split('_')[2]
    if value == 'fr':
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    elif value == 'pd':
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    else:
        raise ValueError(f'There is no handler for value {value}')
    
    step_decision = chain
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('eta_num', passed['el_eta'], m_bins )
    h_den = hist1d('eta_den', total['el_eta'], m_bins )
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)

def make_pt_plot(dataframe, chain, chain_step, l2suffix, value):

    m_bins = np.arange(0, 2000//2, step=50).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    offline = chain.split('_')[2]
    if value == 'fr':
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    elif value == 'pd':
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    else:
        raise ValueError(f'There is no handler for value {value}')
    
    step_decision = chain
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('pt_num', passed['trig_L2_el_pt']/GeV, m_bins )
    h_den = hist1d('pt_den', total['trig_L2_el_pt']/GeV, m_bins )
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)

def make_mu_plot(dataframe, chain, chain_step, l2suffix, value):

    m_bins = [10, 20, 30, 40, 50, 60, 70] # et_bins
    et_cut  = int(chain.split('_')[1][1:])
    offline = chain.split('_')[2]
    if value == 'fr':
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    elif value == 'pd':
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    else:
        raise ValueError(f'There is no handler for value {value}')
    
    step_decision = chain
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('mu_num', passed['avgmu']/MeV, m_bins )
    h_den = hist1d('mu_den', total['avgmu']/MeV, m_bins )
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)

def make_dr_plot(dataframe, chain, chain_step, l2suffix, value):

    m_bins = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.35, 0.40, 0.6]
    et_cut  = int(chain.split('_')[1][1:])
    if value == 'fr':
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    elif value == 'pd':
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
    else:
        raise ValueError(f'There is no handler for value {value}')
    
    step_decision = chain
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('dr_b_num', passed['el_TaP_deltaR'], m_bins )
    h_den = hist1d('dr_b_den', total['el_TaP_deltaR'], m_bins )
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)

def joint_plot(data, x, y, et_cut=None, ylim=None,
                xlim=None, xlabel=None, ylabel=None, data_label=None, criterion=None,
                title=None, cmap=None, marg_color=None, figsize=None):
    if (data_label is not None) and (data_label != 'all') and (criterion is not None):
        data_labeling_func = utils.LABEL_UTILITIES[data_label]
        data = data[data_labeling_func(data, f'el_lh{criterion}')]
    if et_cut is not None:
        data = data[data[var_infos['et']['l2_calo_col']] < et_cut]
    jplot = sns.jointplot(data=data, x=x, y=y, kind='hist', 
                  marginal_kws=dict(thresh=0, color=marg_color),
                  marginal_ticks=True, xlim=xlim, ylim=ylim,
                  joint_kws=dict(thresh=0, cmap=cmap, cbar=True, cbar_kws=dict(orientation="vertical")))
    
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = jplot.ax_joint.get_position()
    pos_marg_x_ax = jplot.ax_marg_x.get_position()
    pos_marg_y_ax = jplot.ax_marg_y.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    jplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    
    if xlabel is not None:
        jplot.ax_joint.set_xlabel(xlabel, fontsize='small')
    if ylabel is not None:
        jplot.ax_joint.set_ylabel(ylabel, fontsize='small')
    
    jplot.figure.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    jplot.figure.patch.set_facecolor('white')
    jplot.figure.suptitle(title, fontsize='medium')
    if figsize is not None:
        jplot.figure.set_figwidth(figsize[0])
        jplot.figure.set_figheight(figsize[1])
    jplot.figure.text(0.7, 0.9, f'Samples:\n{len(data)}', fontsize='small',
                        verticalalignment='top', wrap=True)
    jplot.figure.tight_layout()
    return jplot

var_infos = {
    'et': {
        'label': 'E_{T} [GeV]',
        'plot_func': make_et_plot,
        'col': 'el_et',
        'l2_calo_col': 'trig_L2_cl_et'
    },
    'pt': {
        'label': 'pT [GeV]',
        'plot_func': make_pt_plot,
        'col': 'trig_L2_el_pt'
    },
    'mu': {
        'label': '< #mu >',
        'plot_func': make_mu_plot,
        'col': 'avgmu'
    },
    'eta': {
        'label': '#eta',
        'plot_func': make_eta_plot,
        'col': 'el_eta',
        'l2_calo_col': 'trig_L2_cl_eta'
    },
    'dr': {
        'label': '\Delta R',
        'plot_func': make_dr_plot,
        'col': 'el_TaP_deltaR'
    },
}


val_label_map = {
    'pd': 'P_{D}',
    'fr': 'F_{R}'
}

def make_plot_fig(data, step, chain_name, trigger_strategies, output_dir, var, value, joblib_dump, markers, colors):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try: 
        var_info = var_infos[var]
    except KeyError:
        raise ValueError(f'There is no info for the variable {var}')
    
    trigger = f'{step}_{chain_name}'
    n_strats = len(trigger_strategies)
    energy = chain_name.split("_")[0]
    criterion = chain_name.split("_")[1]
    plot_name = f'{var}_{value}_{step}_{energy}_{criterion}'
    plotpath = os.path.join(output_dir, plot_name)
    colors = COLORS[:n_strats] if colors is None else colors
    markers = MARKERS[:n_strats] if markers is None else markers
    
    root_plots = list()
    for strat in trigger_strategies:
        root_plots.append(var_info['plot_func'](data, trigger.format(strategy=strat), step, strat, value))
    root_plots = np.array(root_plots)

    labels = list()
    for i, trigger_strat in enumerate(trigger_strategies):
        if trigger_strat == 'noringer':
            label_name = 'NoRinger'
        else:
            label_name = ''.join([word.capitalize() for word in trigger_strat.split('_')])
        
        labels.append('%s - %s (%s): %1.2f %%' %(label_name, val_label_map[value], step, root_plots[i, 1]*100))
    
    if joblib_dump:
        joblib.dump(root_plots, plotpath + '.joblib')
        joblib.dump(labels, plotpath + '_labels.joblib')
    
    _, fig = root2fig(root_plots, labels, colors, markers, plot_name)
    
    fig.savefig(plotpath + '.pdf')
    fig.savefig(plotpath + '.png')
    
    return plot_name, fig, labels

def root2fig(root_plots, labels, colors, markers, plot_name):
    var, value, step, energy, criterion = plot_name.split('_')
    fig = rpl.create_canvas(plot_name, canw=1400, canh=1000)
    fig = rpl.plot_profiles(root_plots[:,0], var_infos[var]['label'], colors, markers)
    rpl.format_canvas_axes(YTitleOffset = 0.95)
    add_legend(0.55,0.15, labels)
    rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, energy, criterion), textsize=0.04)
    rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1)
    return plot_name, fig

def cached_root2fig(root_info_dir, plot_name, colors, markers, save=False):

    var, value, step, energy, criterion = plot_name.split('_')
    plotpath = os.path.join(root_info_dir, plot_name)
    root_plots = joblib.load(plotpath + '.joblib')
    labels = joblib.load(plotpath + '_labels.joblib')
    n_strats = len(root_plots)
    colors = COLORS[:n_strats] if colors is None else colors
    markers = MARKERS[:n_strats] if markers is None else markers

    _, fig = root2fig(root_plots, labels, colors, markers, plot_name)

    if save:
        plotpath = os.path.join(root_info_dir, plot_name)
        fig.savefig(plotpath + '.pdf')
        fig.savefig(plotpath + '.png')

    return plot_name, fig, labels