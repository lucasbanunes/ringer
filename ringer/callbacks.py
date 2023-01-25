import logging
from tensorflow import keras


class LoggerCallback(keras.callbacks.Callback):

    def __init__(self, logger_name: str, **kwargs):

        self.logger_name = str(logger_name)
        self.logger = logging.getLogger(self.logger_name)
        self.kwargs = kwargs

    def get_logger_extra(self, logs: dict):
        logs['model-name'] = self.model.name
        if self.kwargs:
            for key, value in self.kwargs.items():
                logs[key] = value
        return logs

    def on_train_begin(self, logs=None):
        self.logger.info('Started training',
                         extra=self.get_logger_extra(logs))

    def on_train_end(self, logs=None):
        self.logger.info('Ended training',
                         extra=self.get_logger_extra(logs))

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.info(f'Epoch {epoch+1}',
                         extra=self.get_logger_extra(logs))

    def on_predict_begin(self, logs=None):
        self.logger.info('Started predicting',
                         extra=self.get_logger_extra(logs))

    def on_predict_end(self, logs=None):
        self.logger.info('Ended predicting',
                         extra=self.get_logger_extra(logs))
