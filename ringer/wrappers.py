from typing import Dict, Any
from tensorflow import keras
from .utils import is_instance
from sklearn.base import BaseEstimator, TransformerMixin


class KerasNeuralNet(BaseEstimator, TransformerMixin):

    def __init__(self, model: keras.Model, compile_kwargs: Dict[str, Any],
                 fit_kwargs: Dict[str, Any], cache_dir: str):

        is_instance(model, keras.Model, 'model')
        self.model = model
        self.compile_kwargs = compile_kwargs
        fit_kwargs_keys = fit_kwargs.keys()
        fit_has_x = 'x' in fit_kwargs_keys
        fit_has_y = 'y' in fit_kwargs_keys
        if fit_has_x or fit_has_y:
            raise ValueError("fit_kwargs can have all parameters with"
                             " exception of parameters \"x\" and \"y\"")
        self.fit_kwargs = fit_kwargs
        self.cache_dir = str(cache_dir)

    def fit(self, X, y):
        backup_callback = keras.callbacks.BackupAndRestore(
            self.cache_dir
        )
        if 'callbacks' in self.fit_kwargs.keys():
            self.fit_kwargs['callbacks'].append(backup_callback)
        else:
            self.fit_kwargs['callbacks'] = [backup_callback]
        self.history_ = self.model.fit(X, y, **self.fit_kwargs)

    def transform(self, X):
        prediction = self.model.predict(X)
        return prediction
