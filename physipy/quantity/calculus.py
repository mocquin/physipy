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


def umap(func, *args, **kwargs):
    """ Extension of python's 'map' function that works with units
    func is the function to map
    args is the arrays on which the function should me mapped
    kwargs must not be used
    """
    if not callable(func):
        raise ValueError("First argument must be a function")

    if len(kwargs) > 0:
        raise ValueError("Keywords arguments are not allowed in this function")

    # Analyse whether arguments are arrays or scalars
    args_len = []
    for i, arg in enumerate(args):
        if np.isscalar(arg):
            args_len.append(1)
        elif isinstance(arg, Quantity) and np.isscalar(arg.value):
            args_len.append(1)
        else:
            shape = np.shape(arg)
            if len(shape) > 1:
                raise NotImplementedError("Only 1D arrays are supported")
            else:
                length = len(arg)
            args_len.append(length)
    ref_length = max(args_len)
    args_modif = list(args)
    for i, arg in enumerate(args):
        if args_len[i] == 1:
            args_modif[i] = np.repeat(args[i], ref_length)
        elif args_len[i] < ref_length:
            raise ValueError("When calling umap: All array/list arguments should have the same length")
        else:
            args_modif[i] = arg
    
    args = tuple(args_modif)
    print("Function arguments:", args)

    out = np.empty((ref_length,), dtype=object) # Declare the output array

    for i in range(ref_length): # iterating on index
        arg = tuple( [arg[i] for arg in args] ) # building tuple of arguments for the current index
        out[i] = func(*arg)

    return out

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

def uvectorize(func, *args, **kwargs):
    """Allow vectorize a function of Quantity.
    
    This function aims to extend numpy.vectorize to Quantity-function.
    
    """
    def func_out(*a, **k):
        return asqarray( umap(func, *a, **k))
    return func_out




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
    """ Vectorize function for functions with one argument, this argument being an N-dimensional np array
    """
    def vec_func(x):
        res = []
        for i in x.flat:
            res.append(func(i))
        res = np.array(res, dtype=object)
        res = asqarray(res)
        res.value = res.value.reshape(x.shape)
        return res
    return vec_func



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
