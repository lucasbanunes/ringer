import os
import logging
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple, Sequence
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

    __BASE_LOGGER_NAME = "jobLogger"

    def __init__(
        self,
        job_id: str,
        dataset: pd.DataFrame,
        target_name: str,
        feature_names: Sequence[str],
        model: keras.Model,
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
        self.dataset = dataset
        self.target_name = target_name
        self.n_folds = n_folds
        self.fold = fold
        self.fold_col_name = fold_col_name

        if feature_names == "all":
            exclude_cols_idx = \
                ~dataset.columns.isin([self.target_name, self.fold_col_name])
            self.feature_names = dataset.columns[exclude_cols_idx]
        else:
            self.feature_names = list(feature_names)

        self.model = model
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.preprocessing_pipeline = preprocessing_pipeline
        self.fit_pipeline = fit_pipeline
        self.output_dir = output_dir
        self.job_dir = os.path.join(output_dir, job_id)
        self.gpu = gpu

        if logger_name:
            self.logger_name = logger_name
        else:
            self.logger_name = self.__BASE_LOGGER_NAME

        self.kwargs = kwargs
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    def run(self):
        """
        Runs the job
        """
        try:
            self.__create_job_dir()
            self.__set_logger()
            logger = self.__get_logger()
            logger.info("Starting job")
            self.dump_inital_params()
            logger.info("Preprocessing dataset")
            x_train, y_train, x_val, y_val = \
                self.preprocess_dataset(self.dataset)
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

    def __create_job_dir(self):
        """Created job_dir"""
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)

    def dump_inital_params(self):
        """Dumps some job data before the fit start"""
        pass

    def __load_dataset(self):
        """Loads the job dataset"""
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
            logger = self.__get_logger()
            logger.info("Fitting pipeline")
            x_train = self.preprocessing_pipeline \
                .fit_transform(dataset.loc[train_idx, self.feature_names])
        else:
            x_train = self.preprocessing_pipeline \
                .transform(dataset.loc[train_idx, self.feature_names])

        y_train = dataset.loc[train_idx, [self.target_name]].values
        del train_idx
        x_val = self.preprocessing_pipeline \
            .transform(dataset.loc[val_idx, self.feature_names])
        y_val = dataset.loc[val_idx, [self.target_name]].values
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
        self.__dump_inital_model(self.model)
        fit_kwargs = self.__get_fit_kwargs_with_extra_callbacks()
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
        self.__dump_history_callback(history)

    def __dump_inital_model(self, model: keras.Model):
        """
        Dumps the model config as a json file and
        the model initial weights as a .h5 file.

        Parameters
        ----------
        model : keras.Model
            Instance of keras model to be trained
        """
        config_path = os.path.join(self.job_dir, "model_config.json")
        with open(config_path, "w") as json_file:
            json_file.write(self.model.to_json(indent=4))

        weights_path = os.path.join(self.job_dir, "initial_weights.h5")
        self.model.save_weights(weights_path)

    def __get_fit_kwargs_with_extra_callbacks(self) -> Dict[str, Any]:
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

    def __dump_history_callback(self, history: keras.callbacks.History):
        """
        Saves the history callback

        Parameters
        ----------
        history : keras.callbacks.History
            History callback returned by keras.Model.fit
        """
        history_filepath = os.path.join(self.job_dir, "history.csv")
        history_df = pd.DataFrame.from_dict(history.history)
        history_df.to_csv(history_filepath)

    def __set_logger(self):
        """
        Creates the job logging.Logger instance
        """
        csv_formatter = logging.Formatter(
            fmt=("%(asctime)s;"
                 "%(levelname)s;"
                 "%(jobId)s"
                 "%(processName)s;"
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
        logger = logging.getLogger(self.__BASE_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.addHandler(csv_handler)
        logger.addHandler(stream_handler)

    def __get_logger(self) -> logging.LoggerAdapter:
        """Returns LoggerAdapter instance with job parameters as extra"""
        logger_extra = dict(jobId=self.job_id)
        logger = logging.getLogger(self.__BASE_LOGGER_NAME)
        adapted_logger = logging.LoggerAdapter(logger, logger_extra)
        return adapted_logger

    @classmethod
    def from_config(cls):
        """
        Creates a new NNFitJob instance from a config file
        """
        raise NotImplementedError
