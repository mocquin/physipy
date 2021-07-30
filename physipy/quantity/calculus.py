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