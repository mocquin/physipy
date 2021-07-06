"""Wrapper of numpy.random distributions

Numpy's random functions cannot be wrapped by array_function/ufunc interface,
so if we want them to be unit-aware, we have to make our own version.

 - https://github.com/numpy/numpy/issues/19382
"""

import numpy as np
from physipy import quantify, Quantity, Dimension


def poisson(lam=1.0, size=None):
    lam = quantify(lam)
    samples = np.random.poisson(lam.value, size=size)
    return Quantity(samples, lam.dimension)

def normal(loc=0.0, scale=1.0, size=None):
    loc = quantify(loc)
    scale = quantify(scale)
    if not loc.dimension == scale.dimension:
        raise DimensionError(loc.dimension, scale.dimension)
    samples = np.random.normal(loc=loc.value, scale=scale.value, size=size)
    return Quantity(samples, loc.dimension)
    
    
def uniform(low=0.0, high=1.0, size=None):
    low = quantify(low)
    high = quantify(high)
    if not low.dimension == high.dimension:
        raise DimensionError(low.dimension, high.dimension)
    samples = np.random.normal(low=low.value, high=high.value, size=size)
    return Quantity(samples, low.dimension)