import numpy as np
from sklearn.base import TransformerMixin

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

class SubGroupPercentageSelector(TransformerMixin):
    
    def __init__(self, selection_dict, percentage=1.):
        is_invalid_percentage = percentage <= 0 or percentage > 1.
        if is_invalid_percentage:
            raise ValueError(f"The percentage parameter must be a value between 0 and 1, not {percentage}")
        self.percentage = percentage
        self.selection_dict = selection_dict
    
    def transform(self, X, y=None):
        selected_idxs = list()
        for group_idx in self.selection_dict.values():
            limit_idx = int(np.floor(len(group_idx)*self.percentage))
            selected_idxs.extend(group_idx[:limit_idx])
        selected = X[:, selected_idxs]
        return selected
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self