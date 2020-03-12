import numpy as np

from .quantity import Quantity, Dimension, DimensionError, dimensionify, quantify, make_quantity



def _iterify(x):
    """make x iterable"""
    return [x] if not isinstance(x, (list, tuple)) else x


def check_dimension(units_in=None, units_out=None):
    """Check dimensions of inputs and ouputs of func"""
    # reading args and making them iterable
    if units_in:
        units_in = _iterify(units_in)
    if units_out:
        units_out = _iterify(units_out)
    
    # define the decorator
    def decorator(func):
        # create a decorated func
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
    """Sets favunit to outputs"""
    # make favunits iterable
    favunits_out = _iterify(favunits_out)
    # make decorator
    def decorator(func):
        # make decorated function
        def decorated_func(*args, **kwargs):
            # compute outputs and iterable it
            ress = _iterify(func(*args, **kwargs))
            # turn outputs to quantity with favunit 
            ress_with_favunit = [make_quantity(res, favunit=favunit) for res, favunit in zip(ress, favunits_out)]
            return tuple(ress_with_favunit) if len(ress_with_favunit) > 1 else ress_with_favunit[0]
        return decorated_func
    return decorator


def dimension_and_favunit(inputs=[], outputs=[]):
    """check dimensions of outputs and inputs, and add favunit to outputs"""
    def decorator(func):
        return set_favunit(outputs)(check_dimension(inputs, outputs)(func))
    return decorator


def convert_to_unit(*unit_in, keep_dim=False):
    """Convert inputs into units - must be same dimension"""
    unit_in = _iterify(unit_in)
    def decorator(func):
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


#### Dropping and adding dimension
def drop_dimension(func):
    """Basically sends the si value to function"""
    def dimension_dropped(*args, **kwargs):
        args = _iterify(args)
        value_args = [quantify(arg).value for arg in args]
        return func(*value_args, **kwargs)
    return dimension_dropped

def add_back_unit_param(*unit_out):
    """Multiply outputs of function by the units_out"""
    unit_out = _iterify(unit_out)
    def decorator(func):
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



def array_to_Q_array(x):
    """Converts an array of Quantity to a Quantity with array value.
    
    First aim to be used with the vectorize.
    
    """
    # aim to deal with np.ndarray
    if type(x) == np.ndarray:
        # if array is size 1
        if x.size == 1:
            return quantify(x.item(0))
        # when size>1 and Quantity
        elif isinstance(x[0], Quantity):
            # extract values into an array
            val_out = np.asarray([qu.value for qu in x])
            # send in a new Quantity
            return Quantity(val_out, 
                            x[0].dimension, 
                            favunit=x[0].favunit)
        # otherwise create a dimless quantity
        else:
            return Quantity(x, Dimension(None))            
    else:
        return quantify(x)
