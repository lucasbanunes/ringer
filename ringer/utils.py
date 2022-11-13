import pandas as pd
import logging
import os
from datetime import datetime
from itertools import product
from typing import Iterable, Dict

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

def medium_keys_mapping(keys:Iterable[str]) -> Dict[str, str]:

    mapping = dict()
    for key in keys:
        medium_key = key.replace('trig_L2_cl', 'L2Calo')
        if medium_key.startswith('el_'):
            medium_key = medium_key[3:]
        mapping[medium_key] = key
    
    return mapping
