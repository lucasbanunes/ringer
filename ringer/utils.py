import logging
import os
from datetime import datetime
from itertools import product
from typing import Dict, Iterable, Union, Tuple
from numbers import Number

import numpy as np
import numpy.typing as npt
import pandas as pd


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

def euclidean_triangle_angle(a: float, b: float, c: float,
    a_err: float=0.0, b_err: float=0.0,
    c_err: float=0.0) -> Tuple[np.float64, np.float64]:
    """
    Returns the angle of vertex A of a triangle given 
    the length of its 3 edges based on the cossine law:
    a^2 = b^2 + a^2 - 2*a*b*cos(alpha)
    alpha = arccos((a^2 - b^2 - c^2)/-2*a*b)

    Parameters
    ----------
    a : float
        Distance from vertex B to C
    b : float
        Distance from vertex A to C
    c : float
        Distance from vertex A to B
    a_err: float 
        Measure error of a
    b_err: float
        Measure error of b
    c_err: float
        Measure error of c

    Returns
    -------
    alpha: numpy.float64
        Vertex A angle in radians
    """
    a2 = a**2
    b2 = b**2
    c2 = c**2
    numerator = (a2) - (b2) - (c2)
    denominator = -2*b*c
    alpha = np.arccos(numerator/denominator)

    alpha_a_err_num = 2*a
    radical = (2*(a2)*(b2))+(2*(c2)*(b2))+(2*(c2)*(a2))
    radical += -(a**4)-(b**4)-(c**4)
    den = np.sqrt(radical)
    alpha_a_err = alpha_a_err_num/den

    alpha_b_err_num = -(a2)-(b2)+(c2)
    alpha_b_err = alpha_b_err_num/(b*den)

    alpha_c_err_num = -(a2)+(b2)-(c2)
    alpha_c_err = alpha_c_err_num/(c*den)

    alpha_err2 = ((alpha_a_err*a_err)**2) + ((alpha_b_err*b_err)**2) + ((alpha_c_err*c_err)**2)
    alpha_err = np.sqrt(alpha_err2)

    return alpha, alpha_err


def get_number_order(num: Number) -> np.integer:
    return np.floor(np.log10(np.abs(num)))  # type: ignore


def significant_around(val: Number,
                      err: Number) -> Tuple[np.floating, np.floating]:
    err_order = get_number_order(err)
    val_arounded = np.around(val, -err_order)  # type: ignore
    err_arounded = np.around(err, -err_order)  # type: ignore
    return val_arounded, err_arounded
