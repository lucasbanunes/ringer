import os
import pandas as pd
import numpy as np
import ringer.utils as utils
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

def joint_plot(data, x, y, et_cut=None, ylim=None,
                xlim=None, xlabel=None, ylabel=None, data_label=None, criterion=None,
                title=None, cmap=None, marg_color=None, figsize=None):
    raise NotImplementedError('Must be refactored')
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

def wasserstein_mapping_plot(distances: pd.Series, title: Union[str, None] = None, 
    legend_kwargs: Union[dict, None] = None, filepath: Union[str, None] = None):
    
    distances = distances.copy()
    fig, ax = plt.subplots(1,1)
    electron = np.array((0,0))
    jet = np.array((electron[0]+distances['el_jet'], 0))
    boosted_x = distances['boosted_el'] if distances['beta'] < np.pi/2 else -distances['boosted_el']
    boosted = np.array((electron[0] + boosted_x, distances['boosted_height']))
    min_xlim = 0-(distances['el_jet']*0.1) if distances['beta'] < np.pi/2 else -distances['boosted_el']*1.1
    max_xlim = max(distances['el_jet'], distances['boosted_el'])*1.1

    # Plotting the triangle first for the markers to be on top
    ax.plot([electron[0], boosted[0], jet[0], electron[0]], 
        [electron[1], boosted[1], jet[1], electron[1]], color='k', alpha=0.3)
    ax.annotate(str(round(distances['boosted_el'], 4)), xy=(electron+boosted)/2)
    ax.annotate(str(round(distances['boosted_jet'], 4)), xy=(jet+boosted)/2)
    ax.annotate(str(round(distances['el_jet'], 4)), xy=(electron+jet)/2)

    # Scatter for vertices marking
    ax.scatter([boosted[0]], [boosted[1]], label='Boosted', color='C0')
    ax.scatter([electron[0]], [electron[1]], label='Electron', color='C1')
    ax.scatter([jet[0]], [jet[1]], label='Jet', color='C2')

    #Plotting reference lines
    ax.plot([-1e3,-1e3],[1e3,1e3], color='red', alpha=0.3, linestyle='--', label='y=x')
    ax.axvline(0, color='blue', alpha=0.3, linestyle='--', label='y=0')

    #Hack for text wrigint on legend
    ax.scatter([1e6], [1e6], label=f'$\\beta = {round(distances["beta"], 2)}$ rad', color='white')

    ax.set(ylim=(0-distances['boosted_height']*0.1,distances['boosted_height']*1.1), 
        xlim=(min_xlim,max_xlim),
        title=title if title is not None else 'Wasserstein Mapping')
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)
    sns.despine(ax=ax)

    if filepath is not None:
        fig.savefig(filepath, transparent=False, facecolor='white')
    
    return fig, ax