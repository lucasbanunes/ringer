from typing import Tuple, Dict
import pandas as pd


class MinMaxScaler():

    def __init__(self, feature_range: Tuple[float, float] = (0., 1.)):
        self.range_min, self.range_max = feature_range

    def fit(self, data: Dict[str, pd.DataFrame]):

        min_values = list()
        max_values = list()
        df_index = list()
        for class_, X in data.items():
            df_index.append(class_)
            min_values.append(X.min())
            max_values.append(X.max())
        idx_rename = {i: name for i, name in enumerate(df_index)}

        self.min_values_ = pd.concat(min_values, axis=1).T
        self.min_values_.rename(idx_rename, axis=0, inplace=True)
        self.min_ = self.min_values_.min()

        self.max_values_ = pd.concat(max_values, axis=1).T
        self.max_values_.rename(idx_rename, axis=0, inplace=True)
        self.max_ = self.max_values_.max()

        return self

    def transform(self, data: Dict[str, pd.DataFrame]):
        transformed_data = dict()
        for class_, X in data.items():
            X_std = (X - self.min_)/(self.max_ - self.min_)
            X_scaled = X_std*(self.range_max - self.range_min)
            X_final = X_scaled + self.range_min
            transformed_data[class_] = X_final

        return transformed_data

    def fit_transform(self, data: Dict[str, pd.DataFrame]):
        return self.fit(data).transform(data)
