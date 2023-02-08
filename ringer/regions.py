import pandas as pd
from typing import Iterable, List
from itertools import product
from ringer.constants import NAMED_ET_ETA_BINS


class EtEtaRegion(object):

    __GEQ_INCLUSIVES = ['left', 'both']
    __LEQ_INCLUSIVES = ['right', 'both']

    def __init__(self, et_range, eta_range, et_idx, eta_idx,
                 et_inclusive, eta_inclusive, et_key, eta_key):
        self.et_range = et_range
        self.eta_range = eta_range
        self.et_idx = et_idx
        self.eta_idx = eta_idx
        self.et_inclusive = et_inclusive
        self.eta_inclusive = eta_inclusive
        self.et_key = et_key
        self.eta_key = eta_key

    def __repr__(self):
        rep = f'EtEtaRegion(et_range={self.et_range}'
        rep += f', eta_range={self.eta_range}'
        rep += f', et_inclusive={self.et_inclusive}'
        rep += f', eta_inclusive={self.eta_inclusive})'
        return rep

    def get_filter(self, data: pd.DataFrame):
        et_filter = data[self.et_key].between(*self.et_range,
                                              inclusive=self.et_inclusive)
        eta_filter = data[self.eta_key] \
            .abs().between(*self.eta_range,
                           inclusive=self.eta_inclusive)
        region_filter = et_filter & eta_filter
        return region_filter

    def get_et_str(self):
        if self.et_inclusive in self.__GEQ_INCLUSIVES:
            left_sign = '<='
        else:
            left_sign = '<'

        if self.et_inclusive in self.__LEQ_INCLUSIVES:
            right_sign = '<='
        else:
            right_sign = '<'

        left_limit, right_limit = self.et_range
        et_str = f'{left_limit} {left_sign} E_T {right_sign} {right_limit}'

        return et_str

    def get_eta_str(self):
        if self.eta_inclusive in self.__GEQ_INCLUSIVES:
            left_sign = '<='
        else:
            left_sign = '<'

        if self.eta_inclusive in self.__LEQ_INCLUSIVES:
            right_sign = '<='
        else:
            right_sign = '<'

        left_limit, right_limit = self.eta_range
        eta_str = f'{left_limit} {left_sign} E_T {right_sign} {right_limit}'

        return eta_str


def get_et_eta_regions(et_bins, eta_bins,
                       et_inclusives, eta_inclusives,
                       et_key, eta_key):
    et_eta_regions = list()
    n_et_bins = len(et_bins)-1
    n_eta_bins = len(eta_bins)-1
    et_eta_idxs = product(range(n_et_bins), range(n_eta_bins))
    for et_eta_idx in et_eta_idxs:
        et_idx, eta_idx = et_eta_idx
        region = EtEtaRegion(
            et_range=et_bins[et_idx: et_idx+2],
            eta_range=eta_bins[eta_idx: eta_idx+2],
            et_idx=et_idx,
            eta_idx=eta_idx,
            et_inclusive=et_inclusives[et_idx],
            eta_inclusive=eta_inclusives[eta_idx],
            et_key=et_key,
            eta_key=eta_key
        )
        et_eta_regions.append(region)
    return et_eta_regions, n_et_bins, n_eta_bins


def get_named_et_eta_regions(region_name: str):
    et_eta_regions, n_et_bins, n_eta_bins = \
        get_et_eta_regions(**NAMED_ET_ETA_BINS[region_name])
    return et_eta_regions, n_et_bins, n_eta_bins


def count_region_samples(
    data_df: pd.DataFrame,
    et_eta_regions: Iterable[EtEtaRegion]
        ) -> List[int]:

    sample_count = list()
    for region in et_eta_regions:
        region_data = data_df.loc[region.get_filter(data_df)]
        sample_count.append(len(region_data))
    return sample_count
