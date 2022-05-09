# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usefull calculus fonctions compatible with Quantity objects.

These are basically numpy function wrapped with dimensions checks.
"""

import numbers as nb
import functools

import numpy as np
import scipy
import scipy.integrate
import scipy.optimize
import sympy as sp

from .dimension import Dimension, DimensionError, SI_UNIT_SYMBOL
from .quantity import quantify, Quantity
from .utils import decorate_with_various_unit, asqarray



#def vectorize(func):
#    """Allow vectorize a function of Quantity.
#    
#    This function aims to extend numpy.vectorize to Quantity-function.
#    
#    """
#    func_vec = np.vectorize(func)
#    def func_Q_vec(*args, **kwargs):
#        res_brute = func_vec(*args, **kwargs)
#        res = asqarray(res_brute)
#        return res
#    return func_Q_vec


def xvectorize(func):
    """
    1-D vectorize func.
    
    func must have signature 'func(arg)', and vectorization is made along arg.
    Returned value will be a Quantity object, even if returned values are 
    dimensionless (because of the use of asqarray).
    
    Just like np.vectorize, this decorator is a utility to wrap a for loop - 
    it does not improve performance in any way.
    """
    def vec_func(x):
        res = []
        for i in x:
            res.append(func(i))
        res = asqarray(res)
        return res
    return vec_func


def ndvectorize(func):
    """
    1-D vectorize func and accept input as ndarray.
    
    func must have signature 'func(arg)', and vectorization is made along arg.
    Returned value will be a Quantity object, even if returned values are 
    dimensionless (because of the use of asqarray).
    
    Basically, func is applied to each value in arg input (as a flat list), 
    and output is reshaped to input shape.
    
    Just like np.vectorize, this decorator is a utility to wrap a for loop - 
    it does not improve performance in any way.
    """
    def vec_func(x):
        res = []
        for i in x.flat:
            res.append(func(i))
        res = asqarray(res)
        res.value = res.value.reshape(x.shape)
        return res
    return vec_func



def vectorize_nd_args(force_quantity=True):
    """
    Vectorize function to any number of scalar or nd arrays. 
    Inputs must be broadcastable.
    
    Only works for single scalar output function.
        
    If force_quantity is True, returned value will always be a Quantity object, 
    even if the function returns dimensionless scalars (floats or likes).
    
    Examples : 
    ==========
    @vectorize_1d_args
    def addition(n, offset):
        if n>offset:
            return n + n + offset
        else:
            return n+offset

    # 1d and scalar
    numbers = (1, 2, 3, 4)*m
    offset = 3*m
    result = addition(numbers, offset)
    
    # both 1d (must be same length)
    numbers = (1, 2, 3, 4)*m
    offset = (2, 5, 1, 5)*m
    result = addition(numbers, offset)
    
    # on 2D with same shape
    img = np.linspace(-1.5, 1.5, num=25).reshape(5,5)*m
    gain = np.ones((5,5))
    offs = np.ones((5,5))*0.5*m

    def apply_nuc(img, gain, offs, sat_min=-1.5*m, sat_max=1.5*m):
        res = img*gain+offs
        if res > sat_max:
            return sat_max
        elif res < sat_min:
            return sat_min
        else:
            return res

    vec_apply_nuc = vectorize_nd_args()(apply_nuc)
    all_2d = vec_apply_nuc(img, gain, offs)
    one_2d = vec_apply_nuc(img, 1, 0.5*m)
    """
    def decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            # first broadcast args
            args = np.broadcast_arrays(*args) 
            flat_args = [arg.flat for arg in args]
            res_iterator = map(func, *flat_args, **kwargs)
            res = [r for r in res_iterator]
            # wrap units
            if force_quantity:
                return asqarray(res).reshape(args[0].shape)
            else:
                return np.asarray(res).reshape(args[0].shape)
        return decorated
    return decorator


def trapz2(Zs, ech_x, ech_y):
    """
    2D integral based on trapz.
    ech_x is horizontal sampling, along row
    ech_y is vertical sampling, along column
    
    
    Example : 
    ---------
        #sample a 2 squared meter, in both direction with different spacing
        nx = 12
        ny = 30
        ech_dx = np.linspace(0*m, 2*m, num=nx)
        ech_dy = np.linspace(0*m, 1*m ,num=ny)
        X, Y = np.meshgrid(ech_dx, ech_dy)
        # make a uniform ponderation
        Zs = np.ones_like(X)
        print(trapz2(Zs, ech_dx, ech_dy))
        #prints 2 m**2
    
    """
    int_x = np.trapz(Zs, axis=-1, x=ech_x)
    int_xy = np.trapz(int_x, axis=-1, x=ech_y)
    return int_xy

def main():
    pass


if __name__ == "__main__":
    main()