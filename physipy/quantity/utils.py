import numpy as np

from .quantity import Quantity, Dimension, DimensionError, dimensionify, quantify, make_quantity



def _iterify(x):
    return [x] if not isinstance(x, (list, tuple)) else x


def check_dimension(units_in=None, units_out=None):
    """Check dimensions of inputs and ouputs of func"""
    if units_in:
        units_in = _iterify(units_in)
    if units_out:
        units_out = _iterify(units_out)
    def decorator(func):
        def decorated_func(*args, **kwargs):
            
            # Checking dimension of inputs
            args = _iterify(args)
            if units_in:
                for arg, unit_in in zip(args, units_in):
                    dim_check_in = dimensionify(unit_in)
                    dim_arg = dimensionify(arg)
                    if not dim_arg == dim_check_in:
                        raise DimensionError(dim_arg, dim_check_in)
            
            # Making outputs iterable
            ress = _iterify(func(*args, **kwargs))

            # Checking dimension of outputs
            if units_out:
                for res, unit_out in zip(ress, units_out):
                    dim_check_out = dimensionify(unit_out)
                    dim_res = dimensionify(res)
                    if not dim_res == dim_check_out:
                        raise DimensionError(dim_res, dim_check_out)
            return tuple(ress) if len(ress) > 1 else ress[0]
        return decorated_func
    return decorator


def set_favunit(*favunits_out):
    """Sets favunit to outputs"""
    favunits_out = _iterify(favunits_out)
    def decorator(func):
        def decorated_func(*args, **kwargs):
            ress = _iterify(func(*args, **kwargs))
            ress_with_favunit = [make_quantity(res, favunit=favunit) for res, favunit in zip(ress, favunits_out)]
            return tuple(ress_with_favunit) if len(ress_with_favunit) > 1 else ress_with_favunit[0]
        return decorated_func
    return decorator

def dimension_and_favunit(inputs=[], outputs=[]):
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
        value_args = []
        for arg in args:
            value_args.append(quantify(arg).value)
        return func(*value_args, **kwargs)
    return dimension_dropped

def add_back_unit_param(*unit_out):
    """Multiply outputs of function by the units_out"""
    unit_out = _iterify(unit_out)
    def decorator(func):
        def dimension_added_back_func(*args, **kwargs):
            ress = func(*args, **kwargs)            
            ress = _iterify(ress)
            ress_q = []
            for res, unit in zip(ress, unit_out):
                ress_q.append(res * unit)
            return tuple(ress_q) if isinstance(ress_q, Iterable) else ress_q
        return dimension_added_back_func
    return decorator


def decorate_with_various_unit(inputs=[], ouputs=[]):
    inputs_str = _iterify(inputs)
    outputs_str = _iterify(ouputs)
    def decorator(func):
        def decorated(*args, **kwargs):
            dict_of_units = {}
            list_inputs_value = [] 
            for arg, input_name in zip(args, inputs_str):
                if input_name == "pass":
                    pass
                else:
                    arg = quantify(arg)
                    si_unit = arg._SI_unitary_quantity()
                    list_inputs_value.append(arg.value)
                    if input_name in dict_of_units and (not si_unit == dict_of_units[input_name]):
                        raise DimensionError((arg._SI_unitary_quantity()).dimension, (dict_of_units[input_name]).dimension)
                    else:
                        dict_of_units[input_name] = arg._SI_unitary_quantity()
                        
            list_outputs_units = [eval(out_str, dict_of_units) for out_str in outputs_str]
                        
            res_brute = func(*list_inputs_value, **kwargs)
            
            res_brute = _iterify(res_brute)
            
            res_q = [res * unit for res, unit in zip(res_brute, list_outputs_units)]
                        
            return tuple(res_q) if len(res_q) > 1 else res_q[0]
        return decorated
    return decorator


    

def array_to_Q_array(x):
    """Converts an array of Quantity to a Quanity of array.
    
    First aim to be used with the vectorize.
    
    """
    #if isinstance(x, Quantity):
    #    return x
    #elif type(x) == np.ndarray:
    #    if x.size == 1:
    #        return x.item(0)
    #    if isinstance(x[0], Quantity):
    #        liste_val = []
    #        for qu in x:
    #            liste_val = liste_val + [qu.value]
    #        valeur_ = np.asarray(liste_val)
    #        dimension_ = x[0].dimension
    #        unite_favorite_ = x[0].favunit
    #        return Quantity(valeur_, dimension_, favunit=unite_favorite_)
    #    else:
    #        return Quantity(x, Dimension(None))                          
    #elif isinstance(x, int) or isinstance(x, float):
    #    return x
    #else:
    #    raise TypeError("Vectorizateur : doit être ")
    
    if type(x) == np.ndarray:
        if x.size == 1:
            return quantify(x.item(0))
        elif isinstance(x[0], Quantity):
            liste_val = []
            for qu in x:
                liste_val = liste_val + [qu.value]
            val_out = np.asarray(liste_val)
            dim_out = x[0].dimension
            favunit_out = x[0].favunit
            return Quantity(val_out, 
                            dim_out, 
                            favunit=favunit_out)
        else:
            return Quantity(x, Dimension(None))            
    else:
        return quantify(x)
