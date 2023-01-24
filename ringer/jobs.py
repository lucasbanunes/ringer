from typing import Dict, Any
from sklearn.base import TransformerMixin


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
        dataset: str,
        model_config: str,
        inital_weights: str,
        compile_kwargs: Dict[str, Any],
        fit_kwargs: Dict[str, Any],
        preprocessing_pipeline: TransformerMixin,
        output_dir: str,
        logger_name: str,
        gpu: str
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
        raise NotImplementedError

    def run(self):
        """
        Runs the fitting job
        """
        raise NotImplementedError
