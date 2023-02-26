import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class DataFrameSelector(TransformerMixin):
    """Selects the columns of DataFrame and returns its values as a numpy array"""
    def __init__(self, selection_cols):
        self.selection_cols = np.array(selection_cols)
    
    def transform(self, X: pd.DataFrame, y=None):
        selected = X[self.selection_cols]
        return selected.values
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

class SubGroupSelector(TransformerMixin):

    def __init__(self, selection_dict):
        self.selection_dict = selection_dict
    
    def transform(self, X, y=None):
        selected = [X[:,group_idx]
                    for group_idx in self.selection_dict.values()]
        return selected
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
