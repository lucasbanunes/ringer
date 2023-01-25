import os
import logging
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple, List
from sklearn.base import TransformerMixin
from tensorflow import keras
from .crossval import ColumnKFold
from .callbacks import LoggerCallback


class NNFitJob():
    """
    Represents a neural net fit job. It implements caching in a way that a job
    initialized with the same parameters poining to the same output_dir
    restarts a job where the last one failed.
    Attributes
    ----------
    job_id : str
        job's guid
    dataset : str
        path to the dataset to be loaded
    model_config : str
        json string returned by keras.Model.to_json
    inital_weights : str
        path to the file containg the initial training weights
        for reproductibility
    compile_kwargs : Dict[str, Any]
        Dict with the model compile kwargs
    fit_kwargs : Dict[str, Any]
        Dict with the model fit kwargs with exception of the validation
        and training data
    preprocessing_pipeline : TransformerMixin
        A transformer according to the scikit learn rules
        IT IS NOT FITTED DURING THE JOB
    output_dir: str
        Directory path to dump job results
    logger_name: str
        logging.Logger name instance to log the results
    gpu: str
        GPU id in which the job will run
    """

    def __init__(
        self,
        job_id: str,
        dataset: Dict[str, Any],
        model_config: str,
        initial_weights: List[NDArray[np.floating]],
        compile_kwargs: Dict[str, Any],
        fit_kwargs: Dict[str, Any],
        preprocessing_pipeline: TransformerMixin,
        n_folds: int,
        fold: int,
        fold_col_name: str,
        output_dir: str,
        logger_name: str,
        gpu: str = "0",
        **kwargs
    ):
        """
        Parameters
        ----------
        job_id : str
            job's guid
        dataset : str
            path to the dataset to be loaded
        model_config : str
            json string returned by keras.Model.to_json
        initial_weights : str
            path to the file containg the initial training weights
            for reproductibility
        compile_kwargs : Dict[str, Any]
            Dict with the model compile kwargs
        fit_kwargs : Dict[str, Any]
            Dict with the model fit kwargs with exception of the validation
            and training data
        preprocessing_pipeline : TransformerMixin
            A transformer according to the scikit learn rules
            IT IS NOT FITTED DURING THE JOB
        output_dir: str
            Directory path to dump job results
        logger_name: str
            logging.Logger name instance to log the results
        gpu: str
            GPU id in which the job will run
        """
        self.job_id = job_id
        self.dataseet = dataset
        self.model_config = model_config
        self.initial_weights = initial_weights
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.preprocessing_pipeline = preprocessing_pipeline
        self.n_folds = n_folds
        self.fold = fold
        self.fold_col_name = fold_col_name
        self.output_dir = output_dir
        self.job_dir = os.path.join(output_dir, job_id)
        self.logger_name = logger_name
        self.gpu = gpu
        self.kwargs = kwargs
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    def run(self):
        """
        Runs the job
        """
        try:
            logger = logging.getLogger(self.logger_name)
            logger.info("Starting job")
            self.create_job_dir()
            self.dump_inital_params()
            logger.info("loading dataset")
            dataset = self.load_dataset()
            logger.info("Loaded dataset")
            logger.info("Preprocessing dataset")
            x_train, y_train, x_val, y_val = self.preprocess_dataset(dataset)
            logger.info("Preprocessed dataset")
            logger.info("Fitting model")
            model_history = self.fit_model(x_train, y_train, x_val, y_val)
            logger.info("Fitted model")
            logger.info("Dumping job results")
            self.dump_results(model_history)
            logger.info("Finished execution")
        except Exception as e:
            logger.exception("An error occured")
            raise e

    def create_job_dir(self):
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)

    def dump_inital_params(self):
        pass

    def load_dataset(self):
        dataset = self.dataset.copy()
        dataset_path = dataset.pop("path")
        dataset_type = dataset.pop("type")
        if dataset_type == "parquet":
            dataset_df = pd.read_parquet(dataset_path, **dataset)
            return dataset_df
        else:
            dataset_str = "\n".join([
                f"{key}: {value}"
                for key, value in self.dataset.items()
            ])
            raise NotImplementedError(
                "Dataset support for this dataset has not been implemented. "
                "Dataset:\n"
                f"{dataset_str}"
            )

    def preprocess_dataset(
        self,
        dataset: pd.DataFrame
    ) -> Tuple[NDArray[np.floating], np.ndarray[np.floating]]:
        crossval = ColumnKFold(self.n_folds, self.fold_col_name)
        val_idx = crossval.get_test_idx(dataset, self.fold)
        train_idx = ~val_idx
        dataset = dataset.drop(self.fold_col_name, axis="columns")
        x_train, y_train = self.preprocessing_pipeline \
            .transform(dataset[train_idx])
        x_val, y_val = self.preprocessing_pipeline \
            .transform(dataset[val_idx])
        return x_train, y_train, x_val, y_val

    def fit_model(self, x_train, y_train, x_val, y_val):
        model = keras.models.model_from_json(self.model_config)
        model.set_weights(self.inital_weights)
        self.__dump_inital_model(model)
        fit_kwargs = self.__get_fit_kwargs_with_extra_callbacks(self)
        model.compile(**self.compile_kwargs)
        model_history = model.fit(x=x_train, y=y_train,
                                  validation_data=(x_val, y_val),
                                  **fit_kwargs)
        self.__dump_history_callback(model_history)
        return model_history

    def __dump_inital_model(self, model: keras.Model):
        config_path = os.path.join(self.job_id, "model_config.json")
        with open(config_path, "w") as json_file:
            json_file.write(self.model_config)

        weights_path = os.path.join(self.job_id, "initial_weights.h5")
        model.save_weights(weights_path)

    def __get_fit_kwargs_with_extra_callbacks(self) -> Dict[str, Any]:
        fit_kwargs = self.fit_kwargs.copy()
        try:
            fit_kwargs["callbacks"] = fit_kwargs["callbacks"].copy()
        except KeyError:
            fit_kwargs["callbacks"] = list()

        backup_callback = keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(self.job_id, "checkpoints"),
            save_freq="epoch",
            delete_checkpoint=False
        )
        logger_callback = LoggerCallback(
            logger_name=self.logger_name,
            job_id=self.job_id
        )
        fit_kwargs["callbacks"].append(backup_callback)
        fit_kwargs["callbacks"].append(logger_callback)
        return fit_kwargs

    def __dump_history_callback(self, history: keras.callbacks.History):

        history_filepath = os.path.join(self.job_dir, "history.csv")
        history_df = pd.DataFrame.from_dict(history.history)
        history_df.to_csv(history_filepath)
