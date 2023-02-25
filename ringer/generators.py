import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import TransformerMixin
from ringer.constants import RINGS_LAYERS, RING_COL_NAME, GENERATOR_CONFIGS, RINGS_PER_LAYERS
from ringer.scalers import AbsSumScaler
from ringer.selectors import SubGroupSelector, SubGroupPercentageSelector

class BaseGenerator(TransformerMixin):

    def __init__(self, scaler="energy_sum"):
        if scaler != "energy_sum":
            raise ValueError('Only scaler energy_sum is supported')

    def __call__(self, X):
        if self.is_fitted():
            transformed = self.transform(X)
        else:
            transformed = self.fit_transform(X)
        return transformed

    def parse_scaler(self, scaler_name):
        scaler = AbsSumScaler()
        return scaler
    
    def is_fitted(self):
        try:
            getattr(self, "fitted_")
        except AttributeError:
            return False
        return True

class TransformerGenerator(BaseGenerator):
    
    def __init__(self, transformer):
        self.transformer = transformer
        
    def transform(self, X, y=None):
        return self.transformer.transform(X, y)
    
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        self.fitted_ = True
        return self

class PercentageRingGenerator(BaseGenerator):
    
    def __init__(self, scaler="energy_sum", ring_percentage=1.):
        
        is_invalid_percentage = ring_percentage <= 0 or ring_percentage > 1.
        if is_invalid_percentage:
            raise ValueError('The ring percentage must be between 0 and 1')
        
        super().__init__(scaler)
        self.ring_percentage = ring_percentage
        self.scaler = self.parse_scaler(scaler)
        self.selector = SubGroupPercentageSelector(
            RINGS_PER_LAYERS,
            ring_percentage
        )
        self.layer_config = RINGS_PER_LAYERS

    def transform(self, X, y=None):
        selected = self.selector.transform(X, y)
        scaled = self.scaler.transform(selected, y)
        return scaled
    
    def fit(self, X, y=None):
        selected = self.selector.fit_transform(X, y)
        self.scaler.fit(selected, y)
        self.fitted_ = True
        return self

class RingGeneratorPerLayer(BaseGenerator):
    
    def __init__(self, scaler="energy_sum"):
        
        super().__init__(scaler)
        self.scaler = self.parse_scaler(scaler)
        self.selector = SubGroupSelector(RINGS_PER_LAYERS)
        self.layer_config = RINGS_PER_LAYERS
    
    def transform(self, X, y=None):
        scaled = self.scaler.transform(X, y)
        transformed = self.selector.transform(scaled, y)
        return transformed
    
    def fit(self, X, y=None):
        scaled = self.scaler.fit_transform(X, y)
        self.selector.fit(scaled, y)
        self.fitted_ = True
        return self

# Generators of each ringer
ringer_generators = {
    "v12": PercentageRingGenerator(ring_percentage=0.5),
    "v12.1": PercentageRingGenerator(ring_percentage=0.5),
    "v18": PercentageRingGenerator(ring_percentage=0.75),
    "v19": PercentageRingGenerator(ring_percentage=0.25),
    "vInception2": RingGeneratorPerLayer()
}  