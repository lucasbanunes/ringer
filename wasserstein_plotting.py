import os
import logging
import logging.config
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from ringer.constants import DEFAULT_FIGSIZE, DEFAULT_DPI
plt.style.use(hep.style.ATLAS)
plt.rc('legend',fontsize='large')
plt.rc('axes',labelsize='x-large')
plt.rc('text',usetex='false')
plt.rc('xtick', labelsize='large')
plt.rc('figure', figsize=DEFAULT_FIGSIZE)
plt.rc('figure', dpi=DEFAULT_DPI)
from ringer.plotting import distance_triangle_plot
from ringer.data import load_var_infos
from ringer.constants import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ringer_debug')


def plot_variables(vars_to_plot: Iterable[str],
                   description_start: str, output_dir: str):
    plots = dict()
    for var_name in vars_to_plot:
        logger.info(f'Plotting {var_name}: {description_start}')
        is_fold_data = wass_distances['description'].str \
            .startswith(description_start)
        is_var_data = wass_distances['name'] == var_name

        if description_start == 'fold':
            var_distances = wass_distances.loc[is_fold_data & is_var_data] \
                .describe()
        elif description_start == 'complete':
            var_distances = pd.DataFrame(
                index=['mean', 'std'],
                columns=wass_distances.columns
                )
            var_distances.loc['mean'] = wass_distances \
                .loc[is_fold_data & is_var_data].iloc[0]
            var_distances.loc['std'] = 0
        else:
            raise ValueError(f"{description_start} value for description_start"
                             "is not supported")

        fig, ax = plt.subplots()
        distance_triangle_plot(
            ax=ax,
            a=var_distances.loc['mean', 'boosted_jet'],   # type: ignore
            b=var_distances.loc['mean', 'boosted_el'],    # type: ignore
            c=var_distances.loc['mean', 'el_jet'],    # type: ignore
            a_err=var_distances.loc['std', 'boosted_jet'],  # type: ignore
            b_err=var_distances.loc['std', 'boosted_el'],  # type: ignore
            c_err=var_distances.loc['std', 'el_jet'],  # type: ignore
            A_label='electron',
            B_label='jet',
            C_label='boosted',
            degrees=False,
            title=f'{var_infos.loc[var_name, "label"]} Wasserstein Mapping',
            plot_references=True,
            legend=True,
            legend_kwargs={} if var_name != 'f1' else dict(loc=2),
            text_kwargs=dict(
                x=text_positions[var_name][0],
                y=text_positions[var_name][1],
                s=var_infos.loc[var_name, 'formula'],
                fontsize='x-large',
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          edgecolor='white',
                          alpha=0)
            )
        )
        figpath = os.path.join(
                output_dir,
                f'{var_name}_{description_start}_mapping.png')
        fig.savefig(figpath)
        plots[var_name] = (fig, ax)

    return plots


text_positions = dict(
    reta=(0.4, 0.87),
    eratio=(0.25, 0.87),
    f1=(0.3, 0.87),
    f3=(0.2, 0.87),
    wstot=(0.25, 0.87),
    weta2=(0.17, 0.87),
    rhad=(0.15, 0.87),
    rhad1=(0.15, 0.87),
    rphi=(0.4, 0.87)
)
output_dir = os.path.join('analysis', 'wasserstein_mapping')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

var_infos = load_var_infos()
wass_distances = pd.read_csv(
    os.path.join('..', '..', 'data', 'wass_distances.csv'),
    index_col=0
)
logger.info('Script start')
vars_to_plot = wass_distances['name'].unique()
plot_variables(vars_to_plot, 'fold', output_dir)
plot_variables(vars_to_plot, 'complete', output_dir)
logger.info('Script end')
