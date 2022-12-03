from typing import Dict, Any, Union
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import is_instance


class KerasNeuralNet(BaseEstimator, TransformerMixin):

    def __init__(self, model: keras.Model, compile_kwargs: Dict[str, Any],
                 fit_kwargs: Dict[str, Any], cache_dir: str,
                 logger_name: Union[str, None] = None,
                 logger_extra: Dict[str, any] = {}):

        is_instance(model, keras.Model, 'model')
        self.model = model
        self.compile_kwargs = compile_kwargs
        fit_kwargs_keys = fit_kwargs.keys()
        fit_has_x = 'x' in fit_kwargs_keys
        fit_has_y = 'y' in fit_kwargs_keys
        if fit_has_x or fit_has_y:
            raise ValueError("fit_kwargs can have all keras.Model.fit"
                             " parameters with"
                             " exception of parameters \"x\" and \"y\"")
        self.fit_kwargs = fit_kwargs
        self.cache_dir = str(cache_dir)
        backup_callback = keras.callbacks.BackupAndRestore(
            self.cache_dir
        )
        self.add_callback(backup_callback)
        self.logger_name = str(logger_name)
        self.logger_extra = logger_extra
        if self.logger_name:
            from .callbacks import ExecutionLogger
            log_callback = ExecutionLogger(
                self.logger_name, **self.logger_extra
            )
            self.add_callback(log_callback)

    def fit(self, X, y):
        self.history_ = self.model.fit(X, y, **self.fit_kwargs)

    def transform(self, X):
        prediction = self.model.predict(X)
        return prediction

    def add_callback(self, callback):
        if 'callbacks' in self.fit_kwargs.keys():
            self.fit_kwargs['callbacks'].append(callback)
        else:
            self.fit_kwargs['callbacks'] = [callback]
