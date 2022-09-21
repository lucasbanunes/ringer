import pandas as pd
import logging
import os
from datetime import datetime
from itertools import product

def get_logger(name: str, id=False, stream=True, file=True):
    logger = logging.getLogger(name)
    if logger.hasHandlers():    #Logger already exists
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if file:
        now = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        id = os.get_pid() if id else ''
        log_filename = f'{now}_{id}_{name}.log'
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def get_electron_label(data: pd.DataFrame, criterion: str):
    return (data['target'] == 1) & (data[criterion] == 1)

def get_jet_label(data: pd.DataFrame, criterion: str):
    return (data['target'] != 1) & (data[criterion] != 1)

LABEL_UTILITIES = {
    'electron': get_electron_label,
    'jet': get_jet_label
}
def get_et_eta_regions(et_bins, eta_bins):
    et_eta_regions = list()
    et_eta_idxs = product(range(len(et_bins)-1), range(len(eta_bins)-1))
    for et_eta_idx in et_eta_idxs:
        et_idx, eta_idx = et_eta_idx
        bin_dict = {
            'et_range': et_bins[et_idx: et_idx+2],
            'eta_range': eta_bins[eta_idx: eta_idx+2]
        }
        et_eta_regions.append(bin_dict)
    return et_eta_regions