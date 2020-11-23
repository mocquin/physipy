# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usefull calculus fonctions compatible with Quantity objects.

These are basically numpy function wrapped with dimensions checks.
"""

import numbers as nb

import numpy as np
import scipy
import scipy.integrate
import scipy.optimize
import sympy as sp

from .dimension import Dimension, DimensionError, SI_UNIT_SYMBOL
from .quantity import quantify, Quantity
from .utils import array_to_Q_array, decorate_with_various_unit, asqarray



def vectorize(func):
    """Allow vectorize a function of Quantity.
    
    This function aims to extend numpy.vectorize to Quantity-function.
    
    """
    func_vec = np.vectorize(func)
    def func_Q_vec(*args, **kwargs):
        res_brute = func_vec(*args, **kwargs)
        res = asqarray(res_brute)
        return res
    return func_Q_vec


def xvectorize(func):
    def vec_func(x):
        res = []
        for i in x:
            res.append(func(i))
        res = np.array(res, dtype=object)
        res = asqarray(res)
        return res
    return vec_func


def ndvectorize(func):
    def vec_func(x):
        res = []
        for i in x.flat:
            res.append(func(i))
        res = np.array(res, dtype=object)
        res = asqarray(res)
        res.value = res.value.reshape(x.shape)
        return res
    return vec_func



def quad(func, x0, x1, *oargs, args=(), **kwargs):
    """A wrapper on scipy.integrate.quad : 
         - will check dimensions of x0 and x1 bounds
         - returned value's dimension is infered by calling func(x0)
    """
    # Cast x bounds in Quantity and check dimension
    x0 = quantify(x0)
    x1 = quantify(x1)
    if not x0.dimension == x1.dimension:
        raise DimensionError(x0.dimension, x1.dimension)
    
    # Get output dimension
    res = func(x0, *args)
    res = quantify(res)
    res_dim = res.dimension
    
    # define a float-version for inputs and outputs
    def func_value(x_value, *oargs):
        # cast back in Quantity
        x = Quantity(x_value, x0.dimension)
        # compute Quantity result
        res_raw = func(x, *args)
        raw = quantify(res_raw)
        # return float-value
        return raw.value
    
    # compute integral with float-value version
    quad_value, prec = scipy.integrate.quad(func_value,
                                      x0.value, x1.value,
                                      *oargs, **kwargs)
    # cast back in Quantity with dimension f(x)dx
    return Quantity(quad_value,
                   res_dim * x0.dimension).rm_dim_if_dimless(), prec


def dblquad(func, x0, x1, y0, y1, *oargs, args=(), **kwargs):
    x0 = quantify(x0)
    x1 = quantify(x1)
    y0 = quantify(y0)
    y1 = quantify(y1)
    
    if not x0.dimension == x1.dimension:
        raise DimensionError(x0.dimension, x1.dimension)
    if not y0.dimension == y1.dimension:
        raise DimensionError(y0.dimension, y1.dimension)
    
    res = func(y0, x0, *args)
    res = quantify(res)
    res_dim = res.dimension
    
    def func_value(y_value, x_value, *args):
        x = Quantity(x_value, x0.dimension)
        y = Quantity(y_value, y0.dimension)
        res_raw = func(y, x, *args)
        raw = quantify(res_raw)
        return raw.value
    
    dblquad_value, prec = scipy.integrate.dblquad(func_value,
                                           x0.value, x1.value,
                                           y0.value, y1.value,
                                           *oargs, **kwargs)
    return Quantity(dblquad_value,
                   res_dim * x0.dimension * y0.dimension).rm_dim_if_dimless(), prec


def tplquad(func, x0, x1, y0, y1, z0, z1, *args):
    x0 = quantify(x0)
    x1 = quantify(x1)
    y0 = quantify(y0)
    y1 = quantify(y1)
    z0 = quantify(z0)
    z1 = quantify(z1)
    
    if not x0.dimension == x1.dimension:
        raise DimensionError(x0.dimension, x1.dimension)
    if not y0.dimension == y1.dimension:
        raise DimensionError(y0.dimension, y1.dimension)
    if not z0.dimension == z1.dimension:
        raise DimensionError(z0.dimension, z1.dimension)
    
    res = func(z0, y0, x0, *args)
    res = quantify(res)
    res_dim = res.dimension
    
    def func_value(z_value, y_value,x_value, *args):
        x = Quantity(x_value, x0.dimension)
        y = Quantity(y_value, y0.dimension)
        z = Quantity(z_value, z0.dimension)
        res_raw = func(z, y, x, *args)
        raw = quantify(res_raw)
        return raw.value
    
    tplquad_value, prec = scipy.integrate.tplquad(func_value,
                                           x0.value, x1.value,
                                           y0.value, y1.value,
                                           z0.value, z1.value,
                                           args=args)
    return Quantity(tplquad_value,
                   res_dim * x0.dimension * y0.dimension * z0.dimension).rm_dim_if_dimless(), prec    


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


def main():
    pass


if __name__ == "__main__":
    main()