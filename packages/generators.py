import numpy as np
import pandas as pd
from collections import defaultdict

RING_COL_NAME = 'trig_L2_cl_ring_{ring_num}'
RINGS_LAYERS = {
    'presample': (0,8),
    'em1': (8,72),
    'em2': (72, 80),
    'em3': (80,88),
    'had1': (88,92),
    'had2': (92,96),
    'had3': (96,100),
}

class RingGenerator(object):

    def __init__(self, ring_percentage=1., norm='l1'):
        if ring_percentage <=0 or ring_percentage > 1.:
            raise ValueError('The ring percentage must be between 0 and 1')
        
        if norm != 'l1':
            raise ValueError('Only norm 1 is supported')
        
        self.norm = lambda x: np.abs( x.sum(axis=1) )
        self.ring_percentage = ring_percentage
        selected_rings = list()
        for layer, ring_range in RINGS_LAYERS.items():
            start_ring, end_ring = ring_range
            n_rings = end_ring-start_ring
            real_ring_end = start_ring + int(np.floor(self.ring_percentage*n_rings))
            selected_rings.append(np.arange(start_ring, real_ring_end))
        #The sort is to garantee order in every python version
        self.selected_rings = np.array([RING_COL_NAME.format(ring_num=ring_num) for ring_num in np.sort(np.concatenate(selected_rings, axis=0))])

    def __call__(self, data):
        rings = data[self.selected_rings].values.astype(np.float32)
        rings = self._normalize(rings)
        return [rings]
    
    def _normalize(self, data):
        norms = self.norm(data)
        norms[norms==0] = 1
        return data/norms[:,None]

# Defaults to all rings
default_percentage = lambda : 1.
ring_percentages = defaultdict(default_percentage)
ring_percentages['v12'] = 0.5
ring_percentages['v12.1'] = 0.5
ring_percentages['v18'] = 0.75
ring_percentages['v19'] = 0.25