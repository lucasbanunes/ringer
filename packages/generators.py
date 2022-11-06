import numpy as np
import pandas as pd
from collections import defaultdict
from packages.constants import RINGS_LAYERS, RING_COL_NAME

class RingGenerator(object):
    
    __VALID_NORMS = ['l1']
    __VALID_LAYER_LEVELS = [0,1,2]
    
    def __init__(self, ring_percentage=1., norm='l1', layer_level=0):
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

    def __call__(self, data):
        rings_data = data[self.selected_rings].astype(np.float32)
        normalized_rings_data = self._normalize(rings_data)
        rings = [normalized_rings_data[ring_group] for ring_group in self.layered_rings]
        return rings
    
    def _normalize(self, data):
        norms = self.norm(data)
        norms[norms==0] = 1
        return data/norms
    
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
            return aux
        elif self.layer_level == 2:
            layered_rings = selected_ringes_per_layer
        
        return selected_rings, layered_rings
    
    def __get_rings(self,):
        selected_rings_per_layer = self.__select_rings_per_layer()
        return self.__apply_layer_level(selected_rings_per_layer)

# Defaults to all rings
default_percentage = lambda : 1.
ring_percentages = defaultdict(default_percentage)
ring_percentages['v12'] = 0.5
ring_percentages['v12.1'] = 0.5
ring_percentages['v18'] = 0.75
ring_percentages['v19'] = 0.25