import pandas as pd
from typing import Any, Generator


class ColumnKFold(object):
    """
    ColumnKFold object implements a Stratified KFold cross validation
    method based on a sequential non-negative integer column of a pandas.DataFrame.
    Given a column with sequential integers "id",
    a test fold "i" is comprised of the samples where:
    id % n_folds == i. When using the column as a reference to build the folds,
    we guarantee reproductibility with the dataset.

    Parameters
    ----------
    n_folds: int
        The number of folds
    sequential_col_name: str
        Name of the column that has a unique non-negative integer for each
        sample sequentially.

    Attributes
    ----------
    n_folds: int
    sequential_col_name: str
    """
    def __init__(self, n_folds: int, sequential_col_name: str):
        self.n_folds = int(n_folds)
        self.sequential_col_name = str(sequential_col_name)

    def split(self, X: pd.DataFrame,
              y: Any = None) -> Generator:
        fold_ids = X[self.sequential_col_name] % self.n_folds
        for test_fold in range(self.n_folds):
            test_idx = fold_ids == test_fold
            train_idx = ~test_idx
            yield test_idx, train_idx

    def get_train_idx(self, X: pd.DataFrame, fold_num: int):
        fold_ids = X[self.sequential_col_name] % self.n_folds
        train_idxs = fold_ids != fold_num
        return train_idxs

    def get_test_idx(self, X: pd.DataFrame, fold_num: int):
        fold_ids = X[self.sequential_col_name] % self.n_folds
        test_idxs = fold_ids == fold_num
        return test_idxs
