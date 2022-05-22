import functools

from functools import lru_cache
from operator import attrgetter

import numpy as np
import sympy as sp

from sympy.parsing.sympy_parser import parse_expr

from .quantity import Quantity, Dimension, DimensionError, dimensionify, quantify, make_quantity



def cached_property_depends_on(*args):
    """
    Decorator to cache a property that depends on other attributes.
    This differs from functools.cached_property in that functools.cached_property is made for immutable atributes.
    
    Use on computation-heavy attributes.
    
    From https://stackoverflow.com/questions/48262273/python-bookkeeping-dependencies-in-cached-attributes-that-might-change
    
    Examples :
    ==========
    import time

    class BADTimeConstantRC:
        
        def __init__(self, R, C):
            self.R = R
            self.C = C
            
        @property
        def tau(self):
            print("Slow computation...")
            time.sleep(5)
            return self.R * self.C
        
    class GOODTimeConstantRC:
        
        def __init__(self, R, C):
            self.R = R
            self.C = C
        
        @cached_property_depends_on('R', 'C')
        def tau(self):
            print("Slow computation...")
            time.sleep(5)
            return self.R * self.C
        
        
    from physipy import units
    ohm = units["ohm"]
    Farad = units["F"]
    bad = BADTimeConstantRC(ohm, Farad)
    print("Bad first : ", bad.tau) # This is long the first time...
    print("Bad second : ", bad.tau) # ... but also the second time !
    
    good = GOODTimeConstantRC(ohm, Farad)
    print("Good fisrt : ", good.tau) # This is long the first time...
    print("Good second : ", good.tau) # ... but not the second time since neither R nor C have changed.
    
    
    """
    attrs = attrgetter(*args)
    def decorator(func):
        _cache = lru_cache(maxsize=None)(lambda self, _: func(self))
        def _with_tracked(self):
            return _cache(self, attrs(self))
        return property(_with_tracked, doc=func.__doc__)
    return decorator



def hard_equal(q1, q2):
    """
    Check if value, dimension, and symbols are equal for Quantities.
    
    This differs from standard equality ("==", or "__eq__") in that the 
    symbols msut be equal as well, in additiion to value and dimension.
    
    Parameters
    ----------
    q1, q2 : Quantity
        Quantity objects to check equality for.
        
    Returns
    -------
    bool
        True if Quantity objects are equal, even for symbols.
    """
    try:
        if q1.dimension == q2.dimension:
            return q1.value == q2.value and q1.symbol == q2.symbol# comparing arrays returns array of bool
        else:
            return False
    except Exception as e:
            return False

def hard_favunit(q, units_list):
    """
    Return True if q hard-equals any unit in units_list.
    
    Parameters
    ----------
    q : Quantity
        Quantity object to test if present in the list.
    units_list : list of Quantity
        List of Quantity objects 
        
    Returns
    -------
    bool 
        Weither q is present in units_list
        
    See also
    --------
    hard_equal
    """
    favs = [unit for unit in units_list if hard_equal(q, unit)]
    return True if len(favs)>=1 else False


def _parse_str_to_dic(exp_str):
    """
    Parse a power expression to a dict.
    
    
    Parameters
    ----------
    exp_str : str
        A string containing a valid power expression.
        
    Returns
    -------
    dict
        A dict with keys the string-symbols and values the corresponding exponent.
    
    Example
    -------
    >>> _parse_str_to_dic("m/s**2")
    {"m":1, "s":-2}
    
    See also
    --------
    parse_expr
    """
    parsed = parse_expr(exp_str)
    exp_dic = {str(key):value for key,value in parsed.as_powers_dict().items()}
    return exp_dic


def _exp_dic_to_q(exp_dic, parsing_dict):
    """
    Helper to expand a dict to a quantity using :
        q = u1**v1 * u2**v2 ...
    where "u" are string of units, and "v" power values.
    """
    q = 1
    for key, value in exp_dic.items():
        # parse string to unit
        u = parsing_dict[key]
        # power up and multiply
        q *= u**value
    return q


def expr_to_q(exp_str, parsing_dict):
    """
    Parse a string expression to a quantity.
    """
    exp_dic = _parse_str_to_dic(exp_str)
    q = _exp_dic_to_q(exp_dic, parsing_dict)
    return q

def strunit_array_to_qunit_array(array_like_of_str, parsing_dict):
    """
    Converts an iterable of strings to an iterable of quantities
    
    Example :
    =========
    
    """
    qs = []
    for s in array_like_of_str:
        q = expr_to_q(s, parsing_dict)
        qs.append(q)
    return qs


def qarange(start_or_stop, stop=None, step=None, **kwargs):
    """Wrapper around np.arange
    
    Meant to be used as a replacement of np.arange  when Quantity objects
    are involved.
    
    Parameters
    ----------
    start_or_stop : Quantity
    stop : Quantity, defaults to None.
    step : Quantity, defaults to None.
    kwargs : kwargs are passed directly to np.arange.
    
    Returns
    -------
    Quantity
        Quantity with value an array based on np.arange
    """
    # start_or_stop param
    final_start_or_stop = quantify(start_or_stop)
    in_dim = final_start_or_stop.dimension
    
    qwargs = dict()
    
    # stop param
    if stop is None:
        pass#final_stop = Quantity(1, in_dim)
    else:
        final_stop = quantify(stop)
        if not final_stop.dimension == final_start_or_stop.dimension:
            raise DimensionError(final_start_or_stop.dimension, final_stop.dimension)
        qwargs["stop"] = final_stop.value
    
    # step param
    if step is None:
        pass#final_step = Quantity(0.1, in_dim)
    else:
        final_step = quantify(step)
        if not final_step.dimension == final_start_or_stop.dimension:
            raise DimensionError(final_start_or_stop.dimension, final_step.dimension)
        qwargs["step"] = final_step.value

    # final call
    val = np.arange(final_start_or_stop.value, **qwargs, **kwargs)
    res = Quantity(val, in_dim)
    return res


def _iterify(x):
    """make x iterable"""
    return [x] if not isinstance(x, (list, tuple)) else x

    
def check_dimension(units_in=None, units_out=None):
    """Check dimensions of inputs and ouputs of function.
    
    Will check that all inputs and outputs have the same dimension 
    than the passed units/quantities. Dimensions for inputs and 
    outputs expects a tuple.
    
    Parameters
    ----------
    units_in : quantity_like or tuple of quantity_like
        quantity_like means an Quantity object or a 
        numeric value (that will be treated as dimensionless Quantity).
        The inputs dimension will be checked with the units_in.
        Defaults to None to skip any check.
    units_out : quantity_like or tuple of quantity_like
        quantity_like means an Quantity object or a 
        numeric value (that will be treated as dimensionless Quantity).
        The outputs dimension will be checked with the units_out.
        Default to None to skip any check.
        
    Returns
    -------
    func:
        decorated function with dimension-checked inputs and outputs.
        
    See Also
    --------
    Other decorators (TODO)
    
    
    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a Greek symbol like :math:`\omega` inline.

    Examples (written in doctest format)
    --------
    >>> def add_meter(x): return x + 1*m
    >>> add_meter = check_dimension((m), (m))(add_meter)
    >>> add_meter(1*m)
    2 m
    >>> add_meter(1*s)
    raise DimensionError
    """

    # reading args and making them iterable
    if units_in:
        units_in = _iterify(units_in)
    if units_out:
        units_out = _iterify(units_out)
    
    # define the decorator
    def decorator(func):
        # create a decorated func
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            # Checking dimension of inputs
            args = _iterify(args)
            if units_in:
                for arg, unit_in in zip(args, units_in):
                    # make everything dimensions
                    dim_check_in = dimensionify(unit_in)
                    dim_arg = dimensionify(arg)
                    # and checking dimensions
                    if not dim_arg == dim_check_in:
                        raise DimensionError(dim_arg, dim_check_in)
            
            # Compute outputs and iterify it
            ress = _iterify(func(*args, **kwargs))

            # Checking dimension of outputs
            if units_out:
                for res, unit_out in zip(ress, units_out):
                    # make everythin dimensions
                    dim_check_out = dimensionify(unit_out)
                    dim_res = dimensionify(res)
                    # and checking dimensions
                    if not dim_res == dim_check_out:
                        raise DimensionError(dim_res, dim_check_out)

            # still return funcntion outputs
            return tuple(ress) if len(ress) > 1 else ress[0]
        return decorated_func
    return decorator


def set_favunit(*favunits_out):
    """Sets favunit to outputs.
    
    Sets favunit for the function outputs.
    
    Parameters
    ----------
    favunits_out : quantity_like or tuple of quantity_like
        quantity_like means an Quantity object or a 
        numeric value (that will be treated as dimensionless Quantity).
        `favunits_out` should be a list of Quantity object that have a defined symbol,
        hence making them suitables favunits.
        
    Returns
    -------
    func:
        decorated function with outputs with a favunit.
        
    See Also
    --------
    Other decorators (TODO)

    Examples (written in doctest format)
    --------
    >>> def add_meter(x): return x + 1*m
    >>> add_meter_favmm = set_favunit(mm)(add_meter)
    >>> add_meter(1*m)
    2000 mm
    """
    # make favunits iterable
    favunits_out = _iterify(favunits_out)
    # make decorator
    def decorator(func):
        # make decorated function
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            # compute outputs and iterable it
            ress = _iterify(func(*args, **kwargs))
            # turn outputs to quantity with favunit 
            ress_with_favunit = [make_quantity(res, favunit=favunit) for res, favunit in zip(ress, favunits_out)]
            return tuple(ress_with_favunit) if len(ress_with_favunit) > 1 else ress_with_favunit[0]
        return decorated_func
    return decorator


def dimension_and_favunit(inputs=[], outputs=[]):
    """Check dimensions of outputs and inputs, and add favunit to outputs.
    
    A wrapper of `check_dimension` and `set_favunit` decorators. The inputs will 
    be dimension-checked, and the output will be both dimension-checked and have
    a favunit. The outputs favunit hence must have same dimension than the
    expected outputs.
    
    See Also
    --------
    set_favunit : Decorator to add favunit to outputs.
    check_dimension : Decorator to check dimension of inputs and outputs.
    """
    def decorator(func):
        return set_favunit(outputs)(check_dimension(inputs, outputs)(func))
    return decorator


def convert_to_unit(*unit_in, keep_dim=False):
    """Convert inputs to values in terms of specified units.
    
    Decorator to convert the function's inputs values in terms
    of specified units.
    
    
    Examples (written in doctest format)
    --------
    >>> @convert_to_unit(mm, mm)
        def add_one_mm(x_mm, y_mm):
            "Expects values as floats in mm"
            return x_mm + y_mm + 1
    >>> print(add_one_mm(1.2*m, 2*m))
    2201
    """
    unit_in = _iterify(unit_in)
    def decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            arg_unitless = []
            for arg, unit in zip(args, unit_in):
                if not keep_dim:
                    arg_unitless.append(arg/unit)
                else:
                    arg_unitless.append(Quantity(arg/unit, unit.dimension))
            return func(*arg_unitless, **kwargs)
        return decorated
    return decorator


def drop_dimension(func):
    """Drop the inputs dimension and sends the SI-value.
    
    Decorator to drop inputs dimensions, quantity value is passed
    in terms of SI-value.
    
    
    Examples
    --------
    >>> @drop_dimension
        def sum_length_from_floats(x, y):
           "Expect dimensionless objects"
           return x + y
    >>> print(sum_length_from_floats(1.2*m, 2*m))
    2.2
    """
    @functools.wraps(func)
    def dimension_dropped(*args, **kwargs):
        args = _iterify(args)
        value_args = [quantify(arg).value for arg in args]
        return func(*value_args, **kwargs)
    return dimension_dropped


def add_back_unit_param(*unit_out):
    """Multiply outputs of function by the units_out
    
    @add_back_unit_param(m, s)
    def timed_sum(x_m, y_m):
        time_s = 10
        return x_m + y_m + 1, time_s
    print(timed_sum(1, 2))
    
    """
    unit_out = _iterify(unit_out)
    def decorator(func):
        @functools.wraps(func)
        def dimension_added_back_func(*args, **kwargs):
            ress = _iterify(func(*args, **kwargs))
            # multiply each output by the unit
            ress_q = [res * unit for res, unit in zip(ress, unit_out)]
            return tuple(ress_q) if len(ress_q) > 1 else ress_q[0]
        return dimension_added_back_func
    return decorator


def decorate_with_various_unit(inputs=[], ouputs=[]):
    """
    allow abitrary specification of dimension and unit: 
    @decorate_with_various_unit(("A", "A"), "A")
    def func(x, y):
        return x+y
        
    It will do 2 things : 
        - check that the inputs have coherent units vs each others
        - set the specified unit to the output
    
    TODO : get rid of eval with a expression parser"""
    inputs_str = _iterify(inputs)
    outputs_str = _iterify(ouputs)
    def decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            dict_of_units = {}
            list_inputs_value = [] 
            # loop over function's inputs and decorator's inputs
            for arg, input_name in zip(args, inputs_str): 
                if input_name == "pass":
                    pass
                #
                else:
                    # turn input into quantity
                    arg = quantify(arg)
                    si_unit = arg._SI_unitary_quantity
                    # store input value 
                    list_inputs_value.append(arg.value)
                    # check if input name (=unit or expression) already exists
                    if input_name in dict_of_units and (not si_unit == dict_of_units[input_name]):
                        raise DimensionError((arg._SI_unitary_quantity).dimension,
                                             (dict_of_units[input_name]).dimension)
                    # if input_name is new, add it's unit to dict
                    else:
                        dict_of_units[input_name] = arg._SI_unitary_quantity
            # compute expression using decorator ouputs
            list_outputs_units = [eval(out_str, dict_of_units) for out_str in outputs_str]
            # compute function res on values
            res_brute = _iterify(func(*list_inputs_value, **kwargs))
            # turn back raw outputs into quantities
            res_q = [res * unit for res, unit in zip(res_brute, list_outputs_units)]
            return tuple(res_q) if len(res_q) > 1 else res_q[0]
        return decorated
    return decorator


def composed_decs(*decs):
    """A wrapper to combine multiple decorators"""
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco

def latex_parse_eq(eq):
    """Tests cases :  
     - v = d/t
     - 2piRC
     - 1/(2piRC)
     - use of sqrt, quad
     - parse lmbda, nu, exp
    """
    #if "$" in eq:
    #    return eq
    #if "=" in eq:
    #    left, right = eq.split("=")
    #    res = "=".join([str(sp.latex(sp.sympify(left))), str(sp.latex(sp.sympify(right)))])
    #    return "$" + res + "$"
    #else:
    #    return "$" + str(sp.sympify(sp.latex(eq))) + "$"
    return eq

    
def latex_eq(eqn):
    """add a 'latex' attribute representation (a string most likely)
    to a function"""
    def wrapper(f):
        f.latex = latex_parse_eq(eqn)
        return f
    return wrapper


def name_eq(name):
    """add a 'name' attribute (a string most likely) to a function"""
    def wrapper(f):
        f.name = name
        return f
    return wrapper


from collections.abc import Iterable

def _flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in _flatten(i)]
    else:
        return [x]

def _is_nested(a):
    return any(isinstance(i, list) for i in a)

def _shape(lst):
    def ishape(lst):
        shapes = [ishape(x) if isinstance(x, list) else [] for x in lst]
        shape = shapes[0]
        if shapes.count(shape) != len(shapes):
            raise ValueError('Ragged list')
        shape.append(len(lst))
        return shape
    return tuple(reversed(ishape(lst)))

def _wrap(a):
    if _is_nested(a):
        shape = _shape(a)
        flatten = list(_flatten(a))
        return flatten, shape
    else:
        return a






def asqarray(array_like):
    """The value returned will always be a Quantity with array value"""
    
    # tuple or list
    if isinstance(array_like, list) or isinstance(array_like, tuple):
        # here should test if any is quantity, not the first one
        # also should test if list is flat or nested
        if any(isinstance(i, Quantity) for i in array_like):
            dim = quantify(array_like[0]).dimension
            val_list = []
            for q in array_like:
                q = quantify(q)
                if q.dimension == dim:    
                    val_list.append(q.value)
                    res_val = np.array(val_list)
                else:
                    raise DimensionError(q.dimension, dim)
            return Quantity(res_val, dim)
        elif isinstance(array_like[0], list):
            flat_array, shape = _wrap(array_like)
            q = asqarray(flat_array)
            q.value = q.value.reshape(shape)
            return q
        # list/tuple of non-quantity value
        else:
            return quantify(array_like)
    # object is array
    elif isinstance(array_like, np.ndarray):
        # non mono-element
        if array_like.size > 1:
            # check all value for dim consistency
            if isinstance(array_like[0], Quantity):
                dim = array_like[0].dimension
                shape = array_like.shape
                val_list = []
                for q in array_like:
                    if q.dimension == dim:    
                        val_list.append(q.value)
                        res_val = np.array(val_list)
                    else:
                        raise DimensionError(q.dim, dim)
                return Quantity(res_val.reshape(shape), dim)
            else:
                return quantify(array_like)
        # array is mono element
        else:
            # array has one d
            if len(array_like.shape)==1:
                return quantify(array_like[0])
            else:
                return quantify(array_like)
    # object is already a Quanity
    elif isinstance(array_like, Quantity):
        return array_like
    else:
        raise ValueError("Type {type(array_like)} not supported")
