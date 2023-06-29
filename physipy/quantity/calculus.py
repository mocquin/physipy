# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usefull calculus fonctions compatible with Quantity objects.

These are basically numpy function wrapped with dimensions checks.
"""
from __future__ import annotations
from typing import Callable

import numbers as nb

import numpy as np
import sympy as sp

from .dimension import Dimension, DimensionError, SI_UNIT_SYMBOL
from .quantity import quantify, Quantity
from .utils import decorate_with_various_unit, asqarray


def xvectorize(func: Callable) -> Callable:
    """
    1-D vectorize func.

    func must have signature 'func(arg)', and vectorization is made along arg.
    Returned value will be a Quantity object, even if returned values are
    dimensionless (because of the use of asqarray).

    Just like np.vectorize, this decorator is a utility to wrap a for loop -
    it does not improve performance in any way.

    Parameter
    ---------
    func : callable
        A function of one parameter.

    Returns
    -------
    callable
        Decorated function.
    """
    def vec_func(x):
        res = asqarray([func(i) for i in x])
        return res
    return vec_func


def ndvectorize(func: Callable) -> Callable:
    """
    1-D vectorize func and accept input as ndarray.

    func must have signature 'func(arg)', and vectorization is made along arg.
    Returned value will be a Quantity object, even if returned values are
    dimensionless (because of the use of asqarray).

    Basically, func is applied to each value in arg input (as a flat list),
    and output is reshaped to input shape.

    Just like np.vectorize, this decorator is a utility to wrap a for loop -
    it does not improve performance in any way.

    Parameter
    ---------
    func : callable
        A function of one parameter.

    Returns
    -------
    callable
        Decorated function.
    """
    def vec_func(x):
        res = asqarray([func(i) for i in x.flat])
        res.value = res.value.reshape(x.shape)
        return res
    return vec_func


def trapz2(Zs: Quantity, ech_x: Quantity, ech_y: Quantity) -> Quantity:
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
