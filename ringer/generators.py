import numpy as np
import pandas as pd
from collections import defaultdict
from ringer.constants import RINGS_LAYERS, RING_COL_NAME, GENERATOR_CONFIGS, RINGS_PER_LAYERS
from ringer.scalers import AbsSumScaler
from ringer.selectors import SubGroupSelector

ringer_generators = {
}

class RingGenerator(object):
    
    __VALID_NORMS = ['l1']
    __VALID_LAYER_LEVELS = [0,1,2]
    
    def __init__(self, ring_percentage=1., norm='l1', layer_level=0, norm_by_level:bool=False):
        if ring_percentage <=0 or ring_percentage > 1.:
            raise ValueError('The ring percentage must be between 0 and 1')
        
        if not norm in self.__VALID_NORMS:
            raise ValueError('Only norm 1 is supported')
        
        layer_level=int(layer_level)
        if not layer_level in self.__VALID_LAYER_LEVELS:
            raise ValueError(f'layer_level supports only the following values {self.__LAYER_LEVELS} ans not {layer_level}')
        
        self.norm = lambda df: df.abs().sum(axis=1).to_frame()
        self.ring_percentage = ring_percentage
        self.layer_level = layer_level
        self.selected_rings, self.layered_rings = self.__get_rings()
        self.norm_by_level=bool(norm_by_level)

    def __call__(self, data: pd.DataFrame):
        rings_data = data[self.selected_rings].astype(np.float32)
        if self.norm_by_level:
            rings = [
                self._normalize(rings_data[ring_group]).values
                for ring_group in self.layered_rings
            ]
        else:
            normalized_rings_data = self._normalize(rings_data)
            rings = [
                normalized_rings_data[ring_group].values
                for ring_group in self.layered_rings
            ]
        return rings
    
    def _normalize(self, data):
        norms = self.norm(data)
        norms[norms==0] = 1
        return data/norms.values
    
    def __select_rings_per_layer(self):
        selected_rings_per_layer = list()
        for layer, ring_range in RINGS_LAYERS.items():
            start_ring, end_ring = ring_range
            n_rings = end_ring-start_ring
            real_ring_end = start_ring + int(np.floor(self.ring_percentage*n_rings))
            ring_nums = np.arange(start_ring, real_ring_end)
            ring_names = np.array([RING_COL_NAME.format(ring_num=ring_num) for ring_num in ring_nums])
            selected_rings_per_layer.append(ring_names)
        return selected_rings_per_layer
    
    def __apply_layer_level(self,selected_rings_per_layer):
        
        selected_rings = np.concatenate(selected_rings_per_layer, axis=0)
        if self.layer_level == 0:
            layered_rings = [selected_rings]
        elif self.layer_level == 1:
            layered_rings = [selected_rings_per_layer[0],
                   np.concatenate(selected_rings_per_layer[1:4], axis=0),
                   np.concatenate(selected_rings_per_layer[4:], axis=0)]
        elif self.layer_level == 2:
            layered_rings = selected_rings_per_layer
        
        return selected_rings, layered_rings
    
    def __get_rings(self,):
        selected_rings_per_layer = self.__select_rings_per_layer()
        return self.__apply_layer_level(selected_rings_per_layer)

def get_ringer_generator_by_version(ringer_version:str):
    generators = list()
    for generator_name, generator_config in GENERATOR_CONFIGS[ringer_version].items():
        if generator_name == 'RingGenerator': # Hack, must be improved
            generator.append(RingGenerator(**generator_config))
        else:
            raise ValueError(f'Generator name is not supported {generator_name}')
    return generators

class RingGeneratorPerLayer():
    
    def __init__(self, scaler="energy_sum"):
        
        if scaler != "energy_sum":
            raise ValueError('Only scaler energy_sum is supported')
        
        self.scaler = self.parse_scaler(scaler)
        self.selector = SubGroupSelector(RINGS_PER_LAYERS)
        self.layer_config = RINGS_PER_LAYERS
    
    def __call__(self, X):
        transformed = self.transform(X)
        return transformed
    
    def parse_scaler(self, scaler_name):
        scaler = AbsSumScaler()
        return scaler
    
    def transform(self, X, y=None):
        scaled = self.scaler.transform(X, y)
        transformed = self.selector.transform(scaled, y)
        return transformed
    
    def fit(self, X, y=None):
        scaled = self.scaler.fit_transform(X, y)
        self.selector.fit_transform(scaled, y)
        self.fitted_ = True
        return self
        
ringer_generators = {
    "vInception2": RingGeneratorPerLayer()
}    
# Defaults to all rings
default_percentage = lambda : 1.
ring_percentages = defaultdict(default_percentage)
ring_percentages['v12'] = 0.5
ring_percentages['v12.1'] = 0.5
ring_percentages['v18'] = 0.75
ring_percentages['v19'] = 0.25