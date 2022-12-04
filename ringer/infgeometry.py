from typing import Dict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as wass_dist


def wasserstein_distance(data: Dict[str, pd.DataFrame]):

    all_vars = np.concatenate(
        [df.columns.values for df in data.values()],
        axis=0
    )
    data_cols = np.unique(all_vars)
    data_combinations = list(combinations(data.keys(), 2))
    index = [f'{left}_{right}' for left, right in data_combinations]
    results = pd.DataFrame(
        columns=data_cols,
        index=index
    )
    for var in data_cols:
        for left, right in data_combinations:
            results.loc[f'{left}_{right}', var] \
                = wass_dist(data[left][var], data[right][var])

    return results
