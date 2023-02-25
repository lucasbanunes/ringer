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