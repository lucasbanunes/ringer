from Gaugi.constants import GeV, MeV
import rootplotlib as rpl

import numpy as np
import root_numpy
import ROOT
ROOT.gStyle.SetOptStat(0);
import array
import os
from itertools import product
import joblib

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

    m_bins = np.arange(0, 2000*10**3//2, step=50*10**3).tolist()
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
    
    h_num = hist1d('pt_num', passed['trig_L2_el_pt']/MeV, m_bins )
    h_den = hist1d('pt_den', total['trig_L2_el_pt']/MeV, m_bins )
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

var2plot_func = {
    'et': make_et_plot,
    'pt': make_pt_plot,
    'mu': make_mu_plot,
    'eta': make_eta_plot,
    'dr': make_dr_plot
}

val_label_map = {
    'pd': 'P_{D}',
    'fr': 'F_{R}'
}

def make_plot_fig(data, step, chain_name, trigger_strategies, output_dir, var, value, joblib_dump):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try: 
        plot_func = var2plot_func[var]
    except KeyError:
        raise ValueError(f'There is no plot funcion for the variable {var}')
    
    trigger = f'{step}_{chain_name}'
    n_strats = len(trigger_strategies)
    root_plots = list()
    for strat in trigger_strategies:
        root_plots.append(plot_func(data, trigger.format(strategy=strat), step, strat, value))
    root_plots = np.array(root_plots)

    labels = list()
    for i, trigger_strat in enumerate(trigger_strategies):
        if trigger_strat == 'noringer':
            label_name = 'NoRinger'
        else:
            label_name = ''.join([word.capitalize() for word in trigger_strat.split('_')])
        
        labels.append('%s - %s (%s): %1.2f %%' %(label_name, val_label_map[value], step, root_plots[i, 1]*100))
    
    fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
    fig = rpl.plot_profiles(root_plots[:,0], 'E_{T} [GeV]', COLORS[:n_strats], MARKERS[:n_strats])
    rpl.format_canvas_axes(YTitleOffset = 0.95)
    add_legend(0.55,0.15, labels)
    rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain_name.split('_')[0], chain_name.split('_')[1]), textsize=0.04)
    rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1)
    plot_name = f'{var}_{value}_{step}_{chain_name.split("_")[0]}_{chain_name.split("_")[1]}'
    plotpath = os.path.join(output_dir, plot_name)
    fig.savefig(plotpath + '.pdf')
    fig.savefig(plotpath + '.png')
    if joblib_dump:
        joblib.dump(root_plots, plotpath + '.joblib')
        joblib.dump(labels, plotpath + '_labels.joblib')
    
    return plot_name, fig, labels