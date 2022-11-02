import pandas as pd
from itertools import product
from .constants import NAMED_ET_ETA_BINS

class EtEtaRegion(object):

    def __init__(self, et_range, eta_range, et_idx, eta_idx, 
        inclusive, et_key, eta_key):
        self.et_range = et_range
        self.eta_range = eta_range
        self.et_idx = et_idx
        self.eta_idx = eta_idx
        self.inclusive = inclusive
        self.et_key = et_key
        self.eta_key = eta_key
    
    def __repr__(self):
        rep = f'EtEtaRegion(et_range={self.et_range}, eta_range={self.eta_range})'
        return rep
    
    def get_filter(self, data: pd.DataFrame):
        et_filter = data[self.et_key].between(*self.et_range, inclusive=self.inclusive)
        eta_filter = data[self.eta_key].between(*self.eta_range, inclusive=self.inclusive)
        region_filter = et_filter & eta_filter
        return region_filter

def get_et_eta_regions(et_bins, eta_bins, inclusive, et_key, eta_key):
    et_eta_regions = list()
    n_et_bins = len(et_bins)-1
    n_eta_bins = len(eta_bins)-1
    et_eta_idxs = product(range(n_et_bins), range(n_eta_bins))
    for et_eta_idx in et_eta_idxs:
        et_idx, eta_idx = et_eta_idx
        region = EtEtaRegion(
            et_range = et_bins[et_idx: et_idx+2],
            eta_range = eta_bins[eta_idx: eta_idx+2],
            et_idx = et_idx,
            eta_idx = eta_idx,
            inclusive = inclusive, 
            et_key = et_key, 
            eta_key = eta_key
        )
        et_eta_regions.append(region)
    return et_eta_regions, n_et_bins, n_eta_bins

def get_named_et_eta_regions(region_name:str):
    et_eta_regions, n_et_bins, n_eta_bins = \
        get_et_eta_regions(**NAMED_ET_ETA_BINS[region_name])
    return et_eta_regions, n_et_bins, n_eta_bins