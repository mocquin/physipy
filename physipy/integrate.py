"""
scipy.integrate wrapped functions

See : https://docs.scipy.org/doc/scipy/reference/integrate.html
"""
import numbers as nb
import numpy as np
import scipy
import scipy.integrate

from physipy import quantify, Quantity, Dimension, DimensionError
from physipy.quantity.utils import check_dimension

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


                 
        

def solve_ivp(fun, 
              t_span, 
              y0,
              method='RK45',
              t_eval=None, 
              dense_output=False, events=None, vectorized=False, args=None, **options):
    
    # first, quantify everything that could be quantity
    tstart, tstop = t_span
    t_span = quantify(tstart), quantify(tstop)
    if not t_span[0].dimension == t_span[1].dimension:
        print("error of dimension")
    y0 = [quantify(y) for y in y0]
    if t_eval is not None:
        t_eval = quantify(t_eval)
        

    t_span_value = t_span[0].value, t_span[1].value
    y0_value = [y.value for y in y0]
    if t_eval is not None:
        t_eval_value = t_eval.value
    else:
        t_eval_value = None
    
        
    # second : rewrite everything without units
    def func_value(t_value, y_value):
        # add back the units
        t = Quantity(t_value, t_span[0].dimension)
        y = Quantity(y_value, y0[0].dimension)
        # compute with units
        res_raw = fun(t, y)
        # extract the numerical value
        raw = quantify(res_raw)
        return raw.value
    
    # compute numerical solution
    sol = scipy.integrate.solve_ivp(
        func_value,
        t_span_value,
        y0_value,
        method=method, 
        t_eval=t_eval, 
        dense_output=dense_output,
        events=events,
        vectorized=vectorized, 
        args=args, 
        **options
    )
    
    # "decorate" the solution with units
    sol.t = Quantity(sol.t, t_span[0].dimension)
    sol.y = Quantity(sol.y, y0[0].dimension)
    func_sol = sol.sol
    
    # for some reason the solution accepts 0*s as well as 0
    @check_dimension(t_span[0].dimension)
    def sol_q(t):
        return Quantity(func_sol(t), y0[0].dimension)#/t_span[0].dimension)
    sol.sol = sol_q
    return sol
                   