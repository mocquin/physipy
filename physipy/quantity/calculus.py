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
from .utils import array_to_Q_array, decorate_with_various_unit


# Generiques
#def linspace(Q_1, Q_2, nb_points=100):
#    """Generate a lineary-spaced vector of Quantity.
#    
#    This function aims to extend numpy.linspace to Quantity objects.
#    
#    """
#    Q_1 = quantify(Q_1)
#    Q_2 = quantify(Q_2)
#    if not Q_1.dimension == Q_2.dimension:
#        raise DimensionError(Q_1.dimension, Q_2.dimension)
#    val_out = np.linspace(Q_1.value, Q_2.value, nb_points)
#    dim_out = Q_1.dimension
#    favunit_out = Q_1.favunit
#    return Quantity(val_out,
#                    dim_out,
#                    favunit=favunit_out)#.rm_dim_if_dimless()
linspace = decorate_with_various_unit(("A", "A"), "A")(np.linspace)

#def interp(x, tab_x, tab_y):
#    """Interpolate the value of x in tab_y based on tab_x.
#    
#    This function aims to extend numpy.interp to Quantity.
#    
#    """
#    x = quantify(x)
#    tab_x = quantify(tab_x)
#    tab_y = quantify(tab_y)
#    if not x.dimension == tab_x.dimension:
#        raise DimensionError(x, tab_x)
#    val_interp = np.interp(x.value, tab_x.value, tab_y.value)
#    dim_interp = tab_y.dimension
#    favunit_interp = tab_y.favunit
#    return Quantity(val_interp,
#                    dim_interp,
#                    favunit=favunit_interp)#.rm_dim_if_dimless()
interp = decorate_with_various_unit(("A", "A", "B"), ("B"))(np.interp)



def vectorize(func):
    """Allow vectorize a function of Quantity.
    
    This function aims to extend numpy.vectorize to Quantity-function.
    
    """
    func_vec = np.vectorize(func)
    def func_Q_vec(*args, **kwargs):
        res_brute = func_vec(*args, **kwargs)
        res = array_to_Q_array(res_brute)
        return res
    return func_Q_vec


# Integrate
def trapz(y, x=None, dx=1.0, *args):
    """Starting from an array of quantity.
    x and dx are exclusifs """
    y = quantify(y)
    if isinstance(x,Quantity):
        value_trapz = np.trapz(y.value, x=x.value, *args)
        dim_trapz = y.dimension * x.dimension
    else:
        dx = quantify(dx)
        value_trapz = np.trapz(y.value, x=x, dx=dx.value, *args)
        dim_trapz = y.dimension * dx.dimension
    return Quantity(value_trapz, 
                    dim_trapz).rm_dim_if_dimless()


def integrate_trapz(Q_min, Q_max, Q_func):
    """Integrate Q_func between Q_min and Q_max.
    
    We start by creating a np.linspace vector between the min and max values.
    Then a Quantity vector with this linspace vector and th corresponding 
    dimension is created.
    
    The dimension's are calculted :
        - the function's output dimension : evaluating the function at Q_min,
            giving the dimension of the
        - the integral's output dimension : multipliying the function ouput
            dimension, by the dimension of the integral's starting point.
    
    """
    Q_min = quantify(Q_min)
    Q_max = quantify(Q_max)
    if not Q_min.dimension == Q_max.dimension:
        raise DimensionError(Q_min.dimension, Q_max.dimension)
    ech_x_val = np.linspace(Q_min.value, Q_max.value, 100)
    Q_ech_x = Quantity(ech_x_val, Q_min.dimension)
    Q_func = vectorize(Q_func)
    Q_ech_y = quantify(Q_func(Q_ech_x))  # quantify for dimensionless cases
    dim_in = quantify(Q_func(Q_min)).dimension
    dim_out = dim_in * Q_min.dimension
    integral = np.trapz(Q_ech_y.value, x=ech_x_val)
    return Quantity(integral, dim_out)#.rm_dim_if_dimless()


def quad(func, x0, x1, *args, **kwargs):
    x0 = quantify(x0)
    x1 = quantify(x1)
    
    if not x0.dimension == x1.dimension:
        raise DimensionError(x0.dimension, x1.dimension)
    
    res = func(x0, *args)
    res = quantify(res)
    res_dim = res.dimension
    
    def func_value(x_value, *args):
        x = Quantity(x_value, x0.dimension)
        
        res_raw = func(x, *args)
        raw = quantify(res_raw)
        return raw.value
    
    quad_value, prec = scipy.integrate.quad(func_value,
                                      x0.value, x1.value,
                                      *args, **kwargs)
    
    return Quantity(quad_value,
                   res_dim * x0.dimension).rm_dim_if_dimless(), prec


def dblquad(func, x0, x1, y0, y1, *args):
    x0 = quantify(x0)
    x1 = quantify(x1)
    y0 = quantify(y0)
    y1 = quantify(y1)
    
    if not x0.dimension == x1.dimension:
        raise DimensionError(x0.dimension, x1.dimension)
    if not y0.dimension == y1.dimension:
        raise DimensionError(y0.dimension, y1.dimension)
    
    res = func(y0,x0, *args)
    res = quantify(res)
    res_dim = res.dimension
    
    def func_value(y_value,x_value, *args):
        x = Quantity(x_value, x0.dimension)
        y = Quantity(y_value, y0.dimension)
        res_raw = func(y,x, *args)
        raw = quantify(res_raw)
        return raw.value
    
    dblquad_value, prec = scipy.integrate.dblquad(func_value,
                                           x0.value, x1.value,
                                           y0.value, y1.value,
                                           args=args)
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
def qroot(func_cal, start):
    start_val = start.value
    start_dim = start.dimension
    def func_cal_float(x_float):
        return func_cal(Quantity(x_float,start_dim))
    return Quantity(scipy.optimize.root(func_cal_float, start_val).x[0], start_dim) #♦Quantity(fsolve(func_cal_float, start_val), start_dim)


#def qbrentq(func_cal, start, stop):
#    start_val = start.value
#    stop_val = stop.value
#    start_dim = start.dimension
#    def func_cal_float(x_float):
#        return func_cal(Quantity(x_float,start_dim))
#    return Quantity(scipy.optimize.brentq(func_cal_float, start_val, stop_val), start_dim) #♦Quantity(fsolve(func_cal_float, start_val), start_dim)

def qbrentq(func_cal, target, start, stop):
    start = quantify(start)
    stop = quantify(stop)
    if not start.dimension == stop.dimension:
        raise DimensionError(start.dimension, stop.dimension)
        
    start_val = start.value
    start_dim = start.dimension
    stop_val = stop.value
    
    def func_float(x):
        return quantify(func_cal(Quantity(x, start_dim))).value - target.value
    res = scipy.optimize.brentq(func_float, start_val, stop.value)
    return Quantity(res, start_dim) # Quantity(fsolve(func_cal_float, 


def main():
    pass


if __name__ == "__main__":
    main()