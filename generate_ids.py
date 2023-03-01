import logging
import logging.config
import os
import json
import numpy as np
import pandas as pd
from Gaugi import load as gload
from Gaugi import save as gsave
from argparse import ArgumentParser
from ringer.data import NamedDatasetLoader
from kepler.pandas.readers import load as kload
from sklearn.model_selection import StratifiedKFold
from ringer.regions import get_named_et_eta_regions
from ringer.constants import LOGGING_CONFIG, RANDOM_STATE, NAMED_DATASETS, NAMED_ET_ETA_BINS
np.random.seed(RANDOM_STATE)


N_FOLDS = 10
MEDIUM_DATASET = "data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97"


def parse_args():
    load_choices = ["parquet", "gaugi", "kepler"]
    name_choices = list(NAMED_DATASETS.keys())
    region_choices = list(NAMED_ET_ETA_BINS.keys())
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="base dataset directory",
                        dest="dataset", type=str)
    parser.add_argument("--name", required=True, help="dataset custom name",
                        choices = name_choices, dest="dataset_name", type=str)
    parser.add_argument("--load", required=True, help="Logic to load the dataset",
                        choices=load_choices, dest="load_func", type=str)
    parser.add_argument("--region", required=True, help="Region named used in this dataset",
                        choices=region_choices, dest="region_name", type=str)
    args = parser.parse_args().__dict__
    return args


def load_data_with_func(filepath, load_func):
    if load_func == "gaugi":
        data = gload(filepath)
        data_df = pd.DataFrame(data["data"], columns=data["features"])
        target_df = pd.DataFrame(data["target"], columns=["target"])
        data_df = pd.concat([data_df, target_df], axis=1)
    elif load_func == "kepler":
        data_df = kload(filepath)
    elif load_func == "parquet":
        data_df = pd.read_parquet(filepath)
    else:
        raise ValueError("Available load functions are gload, kload and parquet")
    return data_df


def process_2017_medium(data_df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    region_name_id = f"{region_name}_id"
    crossval = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    n_samples = len(data_df)
    ifold = 0.0
    for train_idx, val_idx in crossval.split(data_df.values, data_df[["target"]].values):
        logger.info(f"Adding fold {ifold}")
        new_ids = np.arange(ifold, n_samples, N_FOLDS, dtype="uint64")
        data_df.loc[val_idx, f"{region_name}_id"] = new_ids
        ifold += 1
    data_df[region_name_id] = data_df[region_name_id].astype("uint64")
    if data_df[region_name_id].isnull().any():
        raise RuntimeError("Region id has nan values")

    medium_col_rename_filepath = os.path.join("data", "medium_col_rename.json")
    with open(medium_col_rename_filepath, "r") as json_file:
        medium_col_rename = json.load(json_file)

    data_df.rename(medium_col_rename, axis=1, inplace=True)
    return data_df


def process_others(data_df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    region_name_id = f"{region_name}_id"
    region_id = np.arange(len(data_df), dtype="uint64")
    np.random.shuffle(region_id)
    data_df[region_name_id] = pd.Series(region_id)
    return data_df


def run(dataset: str, dataset_name: str, load_func: str, region_name: str):
    basepath = os.path.expanduser("~")
    datapath = os.path.join(basepath, "data", dataset)
    filepath = os.path.join(datapath, dataset + "_et{et}_eta{eta}.npz")
    out_datapath =  NAMED_DATASETS[dataset_name]
    if not os.path.exists(out_datapath):
        os.makedirs(out_datapath)
    dataset_loader = NamedDatasetLoader(dataset_name)
    et_eta_regions, n_ets, n_etas = get_named_et_eta_regions(region_name)

    range_start = 0
    data_df = None
    for region in et_eta_regions:
        logger.info(f"Processing (et_idx, eta_idx) {(region.et_idx, region.eta_idx)}")
        del data_df
        data_df = load_data_with_func(filepath.format(et=region.et_idx, eta=region.eta_idx), load_func)
        if data_df.empty:
            raise ValueError("The data is empty")

        if dataset.endswith(MEDIUM_DATASET):
            # This dataset has a predefined kfold split given by RANDOM_STATE
            # and its columns must be changed to follow the same standards
            # used to name the other dataset columns
            logger.info("Processing 2017_medium dataset")
            data_df = process_2017_medium(data_df, region_name)
        else:
            logger.info("Processing dataset")
            data_df = process_others(data_df, region_name)

        data_df["id"] = np.arange(range_start, range_start+len(data_df))
        range_start += len(data_df)

        logger.info(f"Saving (et_idx, eta_idx) {(region.et_idx, region.eta_idx)}")
        dataset_loader.dump_data_df(data_df, et_bin_idx=region.et_idx, eta_bin_idx=region.eta_idx)

if __name__ == "__main__":
    args = parse_args()
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("ringer_debug")
    run(**args)
    logger.info("Finished")

# dataset = "mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2"
# load_func = "kload"
# region_name = "L2Calo_2017"
# dataset_name = "mc16_boosted"
# dataset = "data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97"
# load_func = "gload"
# region_name = "L2Calo_2017"
# dataset = "data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins"
# load_func = "kload"
# region_name = "L2Calo_2017"