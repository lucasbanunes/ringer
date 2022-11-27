import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from typing import Union, Tuple

from ringer.utils import euclidean_triangle_angle, confidence_interval_str


# def joint_plot(data, x, y, et_cut=None, ylim=None, xlim=None,
#                xlabel=None, ylabel=None, data_label=None, criterion=None,
#                title=None, cmap=None, marg_color=None, figsize=None):
#     raise NotImplementedError('Must be refactored')
#     if (data_label is not None) and (data_label != 'all') and
#        (criterion is not None):
#         data_labeling_func = utils.LABEL_UTILITIES[data_label]
#         data = data[data_labeling_func(data, f'el_lh{criterion}')]
#     if et_cut is not None:
#         data = data[data[var_infos['et']['l2_calo_col']] < et_cut]
#     jplot = sns.jointplot(data=data, x=x, y=y, kind='hist'
#                   marginal_kws=dict(thresh=0, color=marg_color),
#                   marginal_ticks=True, xlim=xlim, ylim=ylim,
#                   joint_kws=dict(thresh=0, cmap=cmap, cbar=True,
#                   cbar_kws=dict(orientation="vertical")))

#     plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
#     # get the current positions of the joint ax and the ax for the marginal x
#     pos_joint_ax = jplot.ax_joint.get_position()
#     pos_marg_x_ax = jplot.ax_marg_x.get_position()
#     pos_marg_y_ax = jplot.ax_marg_y.get_position()
#     # reposition the joint ax so it has the same width as the marginal x ax
#     jplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0,
#                                  pos_marg_x_ax.width, pos_joint_ax.height])
#     # reposition the colorbar using new x positions and y positions of the
#       joint ax

#     if xlabel is not None:
#         jplot.ax_joint.set_xlabel(xlabel, fontsize='small')
#     if ylabel is not None:
#         jplot.ax_joint.set_ylabel(ylabel, fontsize='small')

#     jplot.figure.axes[-1].set_position([.83, pos_joint_ax.y0, .07,
#                                         pos_joint_ax.height])
#     jplot.figure.patch.set_facecolor('white')
#     jplot.figure.suptitle(title, fontsize='medium')
#     if figsize is not None:
#         jplot.figure.set_figwidth(figsize[0])
#         jplot.figure.set_figheight(figsize[1])
#     jplot.figure.text(0.7, 0.9, f'Samples:\n{len(data)}', fontsize='small',
#                         verticalalignment='top', wrap=True)
#     jplot.figure.tight_layout()
#     return jplot


def elipse_section(theta_start: float, theta_end: float,
                   a: float, b: float,
                   num: int = 100
                   ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:

    theta = np.linspace(theta_start, theta_end, num)
    x = a*np.cos(theta)
    y = b*np.sin(theta)
    return x, y


def euclidean_triangle_plot(
    a: float,
    b: float,
    c: float,
    a_err: float = 0.0,
    b_err: float = 0.0,
    c_err: float = 0.0,
    A_label: Union[str, None] = None,
    B_label: Union[str, None] = None,
    C_label: Union[str, None] = None,
    degrees: bool = True,
    title: Union[str, None] = None,
    plot_references: bool = True,
    legend: bool = True,
    legend_kwargs: dict = {},
    text_kwargs: dict = {},
    alpha: float = 1.,
    ax: Union[Axes, None] = None
):
    alpha_ang, alpha_ang_err = euclidean_triangle_angle(
        a, b, c,
        a_err, b_err, c_err)  # type: ignore

    if ax is None:
        _, ax = plt.subplots(1, 1)

    A = np.array((0, 0))
    B = np.array((c, 0))
    C = np.array((b*np.cos(alpha_ang), b*np.sin(alpha_ang)))

    coords = np.row_stack([A, B, C])
    min_x = coords[:, 0].min()
    left_xlim = min_x-(c*0.1)

    # Plotting the triangle first for the vetices' markers to be on top
    ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]],
            color='k', alpha=alpha)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    a_anottation = confidence_interval_str(a, a_err, latex=True)
    ax.annotate(a_anottation, xy=list((B+C)/2), ha='left')
    b_anottation = confidence_interval_str(b, b_err, latex=True)
    ax.annotate(b_anottation, xy=list((A+C)/2), ha='left')
    c_anottation = confidence_interval_str(c, c_err, latex=True)
    ax.annotate(c_anottation, xy=list((A+B)/2), ha='center')

    ax.scatter([C[0]], [C[1]], label=C_label, color='C0', alpha=alpha)
    ax.scatter([A[0]], [A[1]], label=A_label, color='C1', alpha=alpha)
    ax.scatter([B[0]], [B[1]], label=B_label, color='C2', alpha=alpha)

    if plot_references:
        x = np.linspace(-1, 1, 100)
        ax.plot(x, -x,
                color='green', alpha=0.3, linestyle='--', label='y=-x')
        ax.axvline(0, color='blue', alpha=0.3, linestyle='--', label='y=0')
        ax.plot(x, x,
                color='red', alpha=0.3, linestyle='--', label='y=x')

    if degrees:
        alpha_ang_err_str = '' if np.isclose(np.rad2deg(alpha_ang_err), 0) \
            else f'$\\pm {round(np.rad2deg(alpha_ang_err), 4)}$'
        alpha_ang_label = f'$\\alpha = {round(np.rad2deg(alpha_ang), 2)}$'
        alpha_ang_label += alpha_ang_err_str
    else:
        alpha_ang_err_str = '' if np.isclose(alpha_ang_err, 0) \
            else f'$\\pm {round(alpha_ang_err, 4)}$'
        alpha_ang_label = f'$\\alpha = {round(alpha_ang, 2)}$'
        alpha_ang_label += alpha_ang_err_str
    ax.scatter([1e6], [1e6], label=alpha_ang_label, color='white')
    # elipse_coords = elipse_section(0, alpha_ang,   # type:ignore
    #                                a=0.05*c, b=0.05*b*np.sin(alpha_ang))
    # ax.plot(*elipse_coords, color='k', alpha=alpha,
    #         linestyle='-', label=alpha_ang_label)

    ax.set(ylim=ylim,
           xlim=(left_xlim, xlim[1]),
           title=title if title is not None else 'Wasserstein Mapping')

    if legend:
        ax.legend(**legend_kwargs)

    if text_kwargs:
        try:
            text_kwargs['transform']
        except KeyError:
            text_kwargs['transform'] = ax.transAxes
        ax.text(**text_kwargs)

    sns.despine(ax=ax)

    return ax
