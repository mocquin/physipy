



import numbers as nb
import numpy as np
import scipy.optimize

from physipy import quantify, Quantity, Dimension, DimensionError


# Generique 
def root(func_cal, start, args=(), **kwargs):
    start = quantify(start)
    start_val = start.value
    start_dim = start.dimension
    def func_cal_float(x_float):
        q = Quantity(x_float,start_dim)
        return func_cal(q, *args)
    res = scipy.optimize.root(func_cal_float, start_val, **kwargs).x[0]
    return Quantity(res, start_dim)


def brentq(func_cal, start, stop, *oargs, args=(), **kwargs):
    start = quantify(start)
    stop = quantify(stop)
    if not start.dimension == stop.dimension:
        raise DimensionError(start.dimension, stop.dimension)
        
    start_val = start.value
    start_dim = start.dimension
    stop_val = stop.value
    
    def func_float(x):
        res = func_cal(Quantity(x, start_dim), *args)
        return quantify(res).value

    res = scipy.optimize.brentq(func_float, start_val, stop.value, *oargs)
    
    return Quantity(res, start_dim)
