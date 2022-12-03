import logging
from typing import Union


class ModelFitJob(object):

    def __init__(
        self,
        model,
        job_id: str,
        gpu: Union[int, None],
        logger_name: str
    ):

        self.model = model
        self.job_id = str(job_id)
        self.gpu = int(gpu)
        self.logger_name = str(logger_name)
        self.extra = {
            'job_id': self.job_id,
            'gpu': self.gpu,
            'logger_name': self.logger_name
        }

    def run(self, X, y):
        job_logger = logging.getLogger(self.logger_name)
        job_logger.info(
            'Job fit start',
            extra=self.extra
        )
        try:
            if self.gpu is None:
                self.model.fit(X, y)
            else:
                import tensorflow as tf
                with tf.device(f'/gpu:{self.gpu}'):
                    self.model.fit(X, y)
        except Exception as e:
            job_logger.exception(
                'An exception occured during the job fit',
                extra=self.extra
            )
            raise e

        job_logger.info(
            'Job fit ended succesfully',
            extra=self.extra
        )

        return self.model

    def dump(self):
        raise NotImplementedError
