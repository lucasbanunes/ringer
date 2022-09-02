import pandas as pd
import logging
import os
from datetime import datetime

def get_logger(name: str, id=False, stream=True, file=True):
    logger = logging.getLogger(name)
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