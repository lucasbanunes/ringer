import os
import uuid
from abc import ABC, abstractmethod
from itertools import product
import json
import logging
import joblib
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple
from sklearn.base import TransformerMixin
from tensorflow import keras
from .crossval import ColumnKFold
from .callbacks import LoggerCallback

BASE_LOGGER_NAME = "jobLogger"


class BaseFitJob(ABC):

    def __init__(self,
                 logger_name: str):
        if logger_name:
            self.logger_name = logger_name
        else:
            self.logger_name = BASE_LOGGER_NAME

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def set_logger(self):
        """
        Creates the job logging.Logger instance
        """
        csv_formatter = logging.Formatter(
            fmt=("%(asctime)s;"
                 "%(levelname)s;"
                 "%(jobId)s"
                 "%(funcName)s;"
                 "%(lineno)d;"
                 "%(relativeCreated)d;"
                 "%(message)s"))
        standard_formatter = logging.Formatter(
            fmt="%(asctime)s - %(jobId)s - [%(levelname)s]: %(message)s"
        )
        csv_handler = logging.FileHandler(
            filename=os.path.join(self.job_dir, "log.csv"),
            mode="w"
        )
        csv_handler.setFormatter(csv_formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(standard_formatter)
        logger = logging.getLogger(BASE_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.addHandler(csv_handler)
        logger.addHandler(stream_handler)
        return logger

    def get_logger(self) -> logging.LoggerAdapter:
        """Returns LoggerAdapter instance with job parameters as extra"""
        logger_extra = dict(jobId=self.job_id)
        logger = logging.getLogger(self.logger_name)
        if len(logger.handlers) == 0:
            logger = self.set_logger()
        adapted_logger = logging.LoggerAdapter(logger, logger_extra)
        return adapted_logger


class NNFitJob(BaseFitJob):
    """
    Represents a neural net fit job. It implements caching in a way that a job
    initialized with the same parameters poining to the same output_dir
    restarts a job where the last one failed.
    Attributes
    ----------
    job_id : str
            job's guid
        dataset : pd.DataFrame
            Dataframe with the data to be used for fit
        target_name : str
            Target column name
        feature_names : Sequence[str]
            Sequence of feature column names
        model : keras.Model
            Instance of keras model to be trained
        compile_kwargs : Dict[str, Any]
            Dict with the model compile kwargs
        fit_kwargs : Dict[str, Any]
            Dict with the model fit kwargs with exception of the validation
            and training data
        preprocessing_pipeline : TransformerMixin
            A transformer according to the scikit learn rules that preprocesses
            data before NNTraining
        fit_pipeline : bool
            If true fits the preprocessing_pipeline before the NN
        n_folds : int
            Number of Kfolds
        fold : int
            Current fold to fit
        fold_col_name : str
            Fold id column name
        output_dir : str
            Job output_dir
            The job creates inside the output_dir a directory with its id
            and dumps data there
        logger_name : str, optional
            name of logging.Logger to be used, by default None.
            When default creates a log.csv file inside de the job_dir
        gpu : str, optional
            GPU to run the job, by default "0". Not supported yet
    """

    def __init__(
        self,
        job_id: str,
        dataset_info: Dict[str, Any],
        model_config: str,
        compile_kwargs: Dict[str, Any],
        fit_kwargs: Dict[str, Any],
        preprocessing_pipeline: TransformerMixin,
        fit_pipeline: bool,
        n_folds: int,
        fold: int,
        fold_col_name: str,
        output_dir: str,
        logger_name: str = None,
        gpu: str = "0",
        **kwargs
    ):
        """
        Parameters
        ----------
        job_id : str
            job's guid
        dataset : pd.DataFrame
            Dataframe with the data to be used for fit
        target_name : str
            Target column name
        feature_names : Sequence[str]
            Sequence of feature column names
        model : keras.Model
            Instace of keras model to be trained
        compile_kwargs : Dict[str, Any]
            Dict with the model compile kwargs
        fit_kwargs : Dict[str, Any]
            Dict with the model fit kwargs with exception of the validation
            and training data
        preprocessing_pipeline : TransformerMixin
            A transformer according to the scikit learn rules that preprocesses
            data before NNTraining
        fit_pipeline : bool
            If true fits the preprocessing_pipeline before the NN
        n_folds : int
            Number of Kfolds
        fold : int
            Current fold to fit
        fold_col_name : str
            Fold id column name
        output_dir : str
            Job output_dir
            The job creates inside the output_dir a directory with its id
            and dumps data there
        logger_name : str, optional
            name of logging.Logger to be used, by default None.
            When default creates a log.csv file inside de the job_dir
        gpu : str, optional
            GPU to run the job, by default "0". Not supported yet
        kwargs:
            Any other parameter to be created as an attribute
        """
        self.job_id = job_id
        self.dataset_info = dataset_info
        self.n_folds = n_folds
        self.fold = fold
        self.fold_col_name = fold_col_name
        self.model_config_path = model_config
        with open(self.model_config_path, "r") as json_file:
            self.model_config = json.load(json_file)
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.preprocessing_pipeline = preprocessing_pipeline
        self.fit_pipeline = fit_pipeline
        self.output_dir = output_dir
        self.job_dir = os.path.join(output_dir, job_id)
        self.gpu = gpu
        super().__init__(logger_name)

        self.kwargs = kwargs
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    def run(self):
        """
        Runs the job
        """
        try:
            self.create_job_dir()
            logger = self.get_logger()
            logger.info("Starting job")
            # self.dump_inital_params()
            logger.info("Loading dataset")
            # self.dataset, self.feature_names, self.target_names = \
            # self.load_dataset()
            logger.info("Preprocessing dataset")
            # x_train, y_train, x_val, y_val = \
            # self.preprocess_dataset(self.dataset)
            logger.info("Preprocessed dataset")
            logger.info("Fitting model")
            # model_history = self.fit_model(x_train, y_train, x_val, y_val)
            logger.info("Fitted model")
            logger.info("Dumping job results")
            # self.dump_results(model_history)
            logger.info("Finished execution")
            return self.job_id, 0
        except Exception as e:
            logger.exception("An error occured")
            raise e

    def dump_inital_params(self):
        """Dumps some job data before the fit start"""
        pass

    def load_dataset(self):
        """Loads the job dataset"""
        dataset_info = self.dataset_info.copy()
        dataset_path = dataset_info.pop("path")
        dataset_type = dataset_info.pop("type")
        feature_names = dataset_info.pop("feature_names")
        target_names = dataset_info.pop("target_names")
        isin_columns = [self.fold_col_name]
        isin_columns.extend(target_names)

        if feature_names == "all":
            columns_to_load = None
        else:
            columns_to_load = isin_columns.copy()
            columns_to_load.extend(feature_names)

        if dataset_type == "parquet":
            dataset_df = pd.read_parquet(dataset_path,
                                         columns=columns_to_load,
                                         **dataset_info)
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

        if feature_names == "all":
            features_selector = ~dataset_df.columns.isin(isin_columns)
            feature_names = dataset_df.columns[features_selector].to_list()

        return dataset_df, feature_names, target_names

    def preprocess_dataset(
        self,
        dataset: pd.DataFrame
    ) -> Tuple[NDArray[np.floating],
               NDArray[np.floating],
               NDArray[np.floating],
               NDArray[np.floating]]:
        """
        Preprocesses the dataset dividing into validation and training sets
        based on the kfold. Selects features and targets and applies the
        preprocessing pipeline to the feature data

        Parameters
        ----------
        dataset : pd.DataFrame
            Job dataset

        Returns
        -------
        x_train: NDArray[np.floating]
            Preprocessed train feature dataset
        y_train: NDArray[np.floating]
            Preprocessed train target dataset
        x_val: NDArray[np.floating]
            Preprocessed val feature dataset
        y_val: NDArray[np.floating]
            Preprocessed val target dataset
        """
        crossval = ColumnKFold(self.n_folds, self.fold_col_name)
        val_idx = crossval.get_test_idx(dataset, self.fold)
        train_idx = ~val_idx

        if self.fit_pipeline:
            logger = self.get_logger()
            logger.info("Fitting pipeline")
            x_train = self.preprocessing_pipeline \
                .fit_transform(dataset.loc[train_idx, self.feature_names])
        else:
            x_train = self.preprocessing_pipeline \
                .transform(dataset.loc[train_idx, self.feature_names])

        y_train = dataset.loc[train_idx, self.target_names].values
        del train_idx
        x_val = self.preprocessing_pipeline \
            .transform(dataset.loc[val_idx, self.feature_names])
        y_val = dataset.loc[val_idx, self.target_names].values
        return x_train, y_train, x_val, y_val

    def fit_model(self,
                  x_train: NDArray[np.floating],
                  y_train: NDArray[np.floating],
                  x_val: NDArray[np.floating],
                  y_val: NDArray[np.floating]) -> keras.callbacks.History:
        """
        Fits the model on the train and validation sets

        Parameters
        ----------
        x_train : NDArray[np.floating]
            Preprocessed train feature dataset
        y_train : NDArray[np.floating]
            Preprocessed train target dataset
        x_val : NDArray[np.floating]
            Preprocessed val feature dataset
        y_val : NDArray[np.floating]
            Preprocessed val target dataset

        Returns
        -------
        keras.callbacks.History
            History callback returned by keras.Model.fit
        """
        self.model = self.load_inital_model()
        fit_kwargs = self.get_fit_kwargs_with_extra_callbacks()
        self.model.compile(**self.compile_kwargs)
        model_history = self.model.fit(x=x_train, y=y_train,
                                       validation_data=(x_val, y_val),
                                       **fit_kwargs)
        return model_history

    def dump_results(self, history: keras.callbacks.History):
        """
        Dumps the job results

        Parameters
        ----------
        history : keras.callbacks.History
            History callback returned by keras.Model.fit
            It is dumped as a csv file called history.csv
        """
        self.dump_history_callback(history)

    def load_inital_model(self):
        """
        Dumps the model config as a json file and
        the model initial weights as a .h5 file.

        Parameters
        ----------
        model : keras.Model
            Instance of keras model to be trained
        """
        model_config_str = json.dumps(self.model_config)
        model = keras.models.model_from_json(model_config_str)
        config_path = os.path.join(self.job_dir, "model_config.json")
        with open(config_path, "w") as json_file:
            json_file.write(model_config_str)

        weights_path = os.path.join(self.job_dir, "initial_weights.h5")
        model.save_weights(weights_path)
        return model

    def get_fit_kwargs_with_extra_callbacks(self) -> Dict[str, Any]:
        """
        Adds to the fit kwargs the BackupAndRestore and the LoggerCallback
        to implement custom logging and fit cache to the job.

        Returns
        -------
        Dict[str, Any]
            Moddified fit_kwargs
        """
        fit_kwargs = self.fit_kwargs.copy()
        try:
            fit_kwargs["callbacks"] = fit_kwargs["callbacks"].copy()
        except KeyError:
            fit_kwargs["callbacks"] = list()

        backup_callback = keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(self.job_dir, "fit_cache"),
            save_freq="epoch",
            delete_checkpoint=False
        )
        logger_callback = LoggerCallback(
            logger_name=self.logger_name,
            jobId=self.job_id
        )
        fit_kwargs["callbacks"].append(backup_callback)
        fit_kwargs["callbacks"].append(logger_callback)
        return fit_kwargs

    def dump_history_callback(self, history: keras.callbacks.History):
        """
        Saves the history callback

        Parameters
        ----------
        history : keras.callbacks.History
            History callback returned by keras.Model.fit
        """
        history_filepath = os.path.join(self.job_dir, "history.csv")
        history_df = pd.DataFrame.from_dict(history.history)
        history_df["epoch"] = np.arange(1, len(history_df)+1)
        history_df.to_csv(history_filepath, index=False)

    def create_job_dir(self):
        """Created job_dir"""
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)

    @classmethod
    def from_config(cls):
        """
        Creates a new NNFitJob instance from a config file
        """
        raise NotImplementedError


class KFoldNNFitJob(BaseFitJob):

    def __init__(
        self,
        dataset_info: Dict[str, Any],
        model_config: str,
        compile_kwargs: Dict[str, Any],
        fit_kwargs: Dict[str, Any],
        preprocessing_pipeline: TransformerMixin,
        fit_pipeline: bool,
        n_folds: int,
        fold_col_name: str,
        n_inits: int,
        output_dir: str,
        n_jobs: int,
        logger_name: str = None,
        gpu: str = "0",
        **kwargs
    ):
        self.dataset_info = dataset_info
        self.model_config = model_config
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.preprocessing_pipeline = preprocessing_pipeline
        self.fit_pipeline = fit_pipeline
        self.n_folds = n_folds
        self.fold_col_name = fold_col_name
        self.n_inits = n_inits
        self.output_dir = output_dir
        self.job_dir = output_dir
        self.gpu = gpu
        self.n_jobs = n_jobs
        super().__init__(logger_name)
        self.job_id = "KFoldNNFitJob"

        self.kwargs = kwargs
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    def create_job_dir(self):
        """Created job_dir"""
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)

    def run(self):
        self.create_job_dir()
        logger = self.get_logger()
        logger.info("Starting job")
        logger.info("Creating subjob params")
        subjob_params = self.create_subjob_params()
        logger.info("Initiating joblib.Parallel")
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            self.call_nn_fit_job(subjob_param)
            for subjob_param in subjob_params
        )
        # results = parallel(
        #     self.call_nn_fit_job(subjob_param)
        #     for subjob_param in subjob_params
        # )
        return results

    def create_subjob_params(self):

        subjob_params = list()
        folds_inits = product(range(self.n_folds), range(self.n_inits))
        for fold, init in folds_inits:
            param_dict = dict(
                    job_id=str(uuid.uuid4()),
                    dataset_info=self.dataset_info,
                    model_config=self.model_config,
                    compile_kwargs=self.compile_kwargs,
                    fit_kwargs=self.fit_kwargs,
                    preprocessing_pipeline=self.preprocessing_pipeline,
                    fit_pipeline=self.fit_pipeline,
                    n_folds=self.n_folds,
                    fold=fold,
                    fold_col_name=self.fold_col_name,
                    output_dir=self.output_dir,
                    init=init
                )
            subjob_params.append(param_dict)
        return subjob_params

    @staticmethod
    def call_nn_fit_job(nn_fit_job_params):
        job = NNFitJob(**nn_fit_job_params)
        res = job.run()
        return res
