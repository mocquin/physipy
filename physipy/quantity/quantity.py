# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""quantity module : allows manipulation of physical quantities.

TODO:
 - [X] : find a better way to include SI units in
 - [ ] : sum() : problem with init of iterator, using radd, needs to start with sum(y_q,Quantity(1,Dimension("L")))
 - [ ] : avec Q_vectorize qui utilise interp, lève une erreur pas claire si on renvoi une valeur dans un array déjà initialisé avec une autre dimension 
 - [ ] : make DISPLAY_DIGITS and EXP_THRESHOLD variable-attribute, not constant
 - [X] : PENDING : make __true_div__ = __div__
 - [ ] : try to factor quantify and make_quantity
 - [X] : add a method for fast-friendly display in another unit
 - [X] : add methods to check dimensions (is_length, is_surface, etc)
 - [ ] : factorize quad and dblquad, and make nquad
 - [X] : make a decorator for trigo functions 
 - [ ] : deal with precision of quads
 - [X] : make a round method
 - [ ] : quick plot method for array value (matplotlib)
 - [ ] : interp and vectorize adds "no-dimension" if needed - homogenize
 - [ ] : init favunit to SI unit if none is passed ?
 - [X] : make cos, sin, etc method for numpy compatibility
 - [ ] : test quantity with complex and fractions.fractions
 - [ ] : float must be commented for solvers to work....
 - [ ] : make default symbol variable name ?
 - [ ] : make favunit defaults to SI ?
 - [ ] : for trigo method, prevent when unit is sr ?
 - [X] : create a Wrong dimension Error, for trigo functions for eg
 - [X] : deal with numpy slicing a[a>1]
 - [ ] : improve integration of eq, ne (ex : assertNotEqual when dealing with arrays)
 - [ ] : when uncertainties is implemented, add an automatic plotting
 - [X] : add a format method --> need a refactor of repr..
 - [X] : add a method to reset favunit ?
 - [ ] : better tests for complex number support
 - [ ] : see if possible to not rely on sympy, numpy and scipy
 - [ ] : improve code for __array_ufunc__
 - [ ] : better tests for Fraction support


PROPOSITIONS/QUESTIONS : 
 - make sum, mean, integrate, is_dimensionless properties IO methods ?
 - add a 0 quantity to radd to allow magic function sum ?
 - should __pow__ be allowed with array, returning an array of quantities 
     (and quantity of array if power is scalar)
 - should mul and rmul be different : 
    - [1,2,3]*m = ([1,2,3])m 
    - m*[1,2,3] = [1m, 2m, 3m] 
 - make Quantity self)converted to numpy : ex : np.cos(q) if q is dimensionless
     'Quantity' object has no attribute 'cos' ??? 
 -  when multiplying or dividing not quanity with quantity, propagate favunit ?
 - should setting dimension be forbiden ? or with a warning ?
 - make a floordiv ?
 - no np.all in comparison for indexing a[a>1], but then np.all is needed in functions verifications
 - should repr precise the favunit and symbol ?
 - switch base system by replacing the dimension dict (or making it setable)
 - exponent repr : use re to change “**” to “^” or to “”
 - base nominal representation and str should allow plain copy/paste to be reused in code
 - list unicode possible changes : micron character, superscripts for exponents
 - quantify in each method is ugly
 - should redefine every numpy ufunc as method ?

"""

import scipy
import scipy.integrate
import scipy.optimize

import numbers as nb
import numpy as np
import sympy as sp

import warnings

from .dimension import Dimension, DimensionError, SI_UNIT_SYMBOL

# # Constantes
UNIT_PREFIX= " "
DISPLAY_DIGITS = 2
EXP_THRESHOLD = 3
UNIT_SUFFIX = ""
#SCIENTIFIC = '%.' + str(DISPLAY_DIGITS) + 'E' # (syntaxe : "%.2f" % mon_nombre
#CLASSIC =  '%.' + str(DISPLAY_DIGITS) + 'f'

HANDLED_FUNCTIONS = {}


class Quantity(object):
    """Quantity class : """
    
    #DIGITS = DISPLAY_DIGITS
    #EXP_THRESH = EXP_THRESHOLD
    
    def __init__(self, value, dimension, symbol="UndefinedSymbol", favunit=None):
        self.__array_priority__ = 100
        self.value = value
        self.dimension = dimension
        self.symbol = symbol
        self.favunit = favunit

    def __setattr__(self, name, value):
        if name == "value":
            if isinstance(value,np.ndarray):
                super().__setattr__(name,np.atleast_1d(value))
                super().__setattr__("size",self.value.size)
            elif (isinstance(value,nb.Number) or type(value) == np.int64 or
                  type(value) == np.int32):
            #isinstance(value, (int, float)):#isinstance(value,float) or 
            #isinstance(value,int) or type(value) == numpy.int64 or 
            #type(value) == numpy.int32: 
            #or numpy.isrealobj(valeur):
                super().__setattr__(name,value)
                super().__setattr__("size",1)
            elif isinstance(value,list):
                super().__setattr__(name,np.array(value))
            elif value is None:
                super().__setattr__(name, value)
            else: 
                raise TypeError(("Value of Quantity must be a number "
                                 "or numpy array, not {}").format(type(value)))
        elif name == "dimension":
            if isinstance(value,Dimension) :
                super().__setattr__(name,value)
            else: 
                raise TypeError(("Dimension of Quantity must be a Dimension,"
                                 "not {}").format(type(value)))
        elif name == "symbol":
            if isinstance(value,str):                       
                super().__setattr__(name,sp.Symbol(value))       
            elif isinstance(value,sp.Expr):                  
                super().__setattr__(name, value)                          
            else: 
                raise TypeError(("Symbol of Quantity must be a string "
                                 "or a sympy-symbol, "
                                 "not {}").format(type(value)))
        elif name == "favunit":
            if isinstance(value,Quantity) or value == None:
                super().__setattr__(name, value)
            elif np.isscalar(value):
                super().__setattr__(name, None)
            else:
                raise TypeError(("Favorite unit of Quantity must be a Quantity "
                                 "or None, not {}").format(type(value)))
        else:
            super().__setattr__(name, value)

    def __add__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension:                                                               
            raise DimensionError(self.dimension, y.dimension)                          
        return Quantity(self.value + y.value,
                        self.dimension)    

    def __radd__(self, x): return self + x

    def __sub__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension: 
            raise DimensionError(self.dimension, y.dimension)
        return Quantity(self.value - y.value,
                        self.dimension)

    def __rsub__(self, x): return quantify(x) - self

    def __mul__(self,y):
        y = quantify(y)
        return Quantity(self.value * y.value, 
                        self.dimension * y.dimension, 
                        symbol = self.symbol * y.symbol).rm_dim_if_dimless() 
    
    __rmul__ = __mul__

    def __truediv__(self, y):
        y = quantify(y)
        return Quantity(self.value / y.value,
                        self.dimension / y.dimension,
                        symbol = self.symbol / y.symbol).rm_dim_if_dimless()

    def __rtruediv__(self, x): return quantify(x) / self

    def __floordiv__(self, y):
        """
        Any returned quantity should be dimensionless, but leaving the 
        Quantity().remove() because more intuitive
        """
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return Quantity(self.value // y.value,
                       self.dimension).rm_dim_if_dimless()
    
    def __mod__(self,y):
        """
        There is no rm_dim_if_dimless() because a 
        modulo operation would not change the dimension.
        
        """
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return Quantity(self.value % y.value,
                        self.dimension)#.rm_dim_if_dimless()

    def __pow__(self,power):
        """
        A power must always be a dimensionless scalar. 
        If a = 1*m, we can't do a ** [1,2], because the result would be
        an array of quantity, and can't be a quantity with array-value, 
        since the quantities won't be the same dimension.
        
        """
        if not np.isscalar(power):#(isinstance(power,int) or isinstance(power,float)):
            raise TypeError(("Power must be a number, "
                            "not {}").format(type(power)))
        return Quantity(self.value ** power, 
                        self.dimension ** power,
                        symbol = self.symbol ** power).rm_dim_if_dimless()

    def __neg__(self): return self * (-1)
    
    def __len__(self): return len(self.value)
    
    def __bool__(self): return bool(self.value)

    def __min__(self):
        return Quantity(min(self.value),
                        self.dimension,
                        favunit=self.favunit)

    def __max__(self):
        return Quantity(max(self.value),
                        self.dimension,
                        favunit=self.favunit)

    def __eq__(self,y):
        y = quantify(y)
        if self.dimension == y.dimension:
            return self.value == y.value # comparing arrays returns array of bool
        else:
            return False

    def __ne__(self,y):
        return np.invert(self == y) #np.invert for element-wise not, for array compatibility
        
    def __gt__(self,y):
        y = quantify(y)
        if self.dimension == y.dimension:
            return self.value > y.value
        else:
            raise DimensionError(self.dimension,y.dimension)

    def __lt__(self,y):
        y = quantify(y)
        if self.dimension == y.dimension:
            return self.value < y.value
        else:
            raise DimensionError(self.dimension,y.dimension)        

    def __ge__(self,y): return (self > y) | (self == y) # or bitwise

    def __le__(self,y): return (self < y) | (self == y) # or bitwise
    
    def __abs__(self):
        return Quantity(abs(self.value),
                        self.dimension,
                        symbol = self.symbol,
                        favunit = self.favunit)

    def __complex__(self):
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, Dimension(None), binary=False)
        return complex(self.value)
            
    def __int__(self):
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, Dimension(None), binary=False)
        return int(self.value)

    def __float__(self):
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, Dimension(None), binary=False)
        return float(self.value)
    
    def __round__(self, i):
        return Quantity(round(self.value, i), 
                       self.dimension,
                       favunit = self.favunit)
    
    def __copy__(self):
        return Quantity(self.value, self.dimension, favunit=self.favunit, symbol=self.symbol)
    
    def __repr__(self):
        return '<Quantity : ' + str(self.value) + " " + str(self.dimension.str_SI_unit()) + ">"        

    def __str__(self):
        complement_value_for_repr = self._compute_complement_value() 
        if not complement_value_for_repr == "":
            return str(self._compute_value()) + UNIT_PREFIX + complement_value_for_repr + UNIT_SUFFIX
        else: 
            return str(self._compute_value()) + UNIT_SUFFIX

    #def __format_raw__(self, format_spec):
    #    return format(self.value, format_spec) + " " + str(self.dimension.str_SI_unit())

    def __format__(self, format_spec):
        complement_value_for_repr = self._compute_complement_value()
        if not complement_value_for_repr == "":
            return format(self._compute_value(), format_spec) + UNIT_PREFIX + complement_value_for_repr + UNIT_SUFFIX
        else: 
            return format(self._compute_value(), format_spec) + UNIT_PREFIX
    
    def _compute_value(self):
        """Return the numerical value corresponding to favunit."""
        if isinstance(self.favunit, Quantity):
            ratio_favunit = make_quantity(self/self.favunit)
            return ratio_favunit.value
        else:
            return self.value
    
    def _compute_complement_value(self, custom_favunit=None):
        """Return the complement to the value as a str."""
        if custom_favunit is None:
            favunit = self.favunit
        else:
            favunit = custom_favunit
        if isinstance(favunit, Quantity):
            ratio_favunit = make_quantity(self/favunit)
            dim_SI = ratio_favunit.dimension
            if dim_SI == Dimension(None):
                return str(favunit.symbol)
            else:
                return str(favunit.symbol) + "*" + dim_SI.str_SI_unit()
        else:
            return self.dimension.str_SI_unit()

    #used for plotting
    @property
    def _SI_unitary_quantity(self):
        """Return a one-value quantity with same dimension.
        
        Such that self = self.value * self._SI_unitary_quantity
        """
        return Quantity(1, self.dimension, symbol=self.dimension.str_SI_unit())

    
    def __getitem__(self, idx):
        if isinstance(self.value, np.ndarray):
            return Quantity(self.value[idx],
                            self.dimension,
                            favunit=self.favunit)
        else:
            raise TypeError("Can't index on non-array value.")

    def __setitem__(self, idx, q):
        q = quantify(q)
        if not q.dimension == self.dimension:
            raise DimensionError(q.dimension,self.dimension)
        if isinstance(idx,np.bool_) and idx == True:
            self.valeur = q.value
        elif isinstance(idx,np.bool_) and idx == False:
            pass
        else:
            self.value[idx] = q.value

    def __iter__(self):
        return QuantityIterator(self)
    
    def std(self, *args, **kwargs):
        return np.std(self, *args, **kwargs)
    
    def inverse(self):
        """is this method usefull ?"""
        return 1. / self

    def sum(self, **kwargs): return np.sum(self, **kwargs)
    
    def mean(self, **kwargs): return np.mean(self, **kwargs)
    
    def integrate(self, **kwargs): return np.trapz(self, **kwargs)
    
    def is_dimensionless(self):
        return self.dimension == Dimension(None)

    def rm_dim_if_dimless(self):
        if self.is_dimensionless():
            return self.value          
        else:                           
            return self
        
    def has_integer_dimension_power(self):
        return all(value == int(value) for value in self.dimension.dim_dict.values())
    
    def to(self, y):
        """return quantity with another favunit."""
        if not isinstance(y, Quantity):
            raise TypeError("Cannot express Quantity in not Quantity")
        q = self.__copy__()
        q.favunit = y
        return q

    def ito(self, y):
        """in-place change of favunit."""
        self.favunit = y
        return self
    
    def into(self, y):
        """like to, but with same dimension"""
        if not self.dimension == y.dimension:
            raise ValueError("must have same unit. Try to().")
        return self.to(y)
        
    def iinto(self, y):
        """like ito, but with same dimension"""
        if not self.dimension == y.dimension:
            raise ValueError("must have same unit. Try to().")
        return self.ito(y)

    # Shortcut for checking dimension
    def is_length(self): return self.dimension == Dimension("L")

    def is_surface(self): return self.dimension == Dimension("L")**2

    def is_volume(self): return self.dimension == Dimension("L")**3

    def is_time(self): return self.dimension == Dimension("T")

    def is_mass(self): return self.dimension == Dimension("M")

    def is_angle(self): return self.dimension == Dimension("RAD")

    def is_solid_angle(self): return self.dimension == Dimension("SR")

    def is_temperature(self): 
        return self.dimension == Dimension("theta")

    def is_dimensionless_ext(self):
        return self.is_dimensionless() or self.is_angle()
    
    def check_dim(self, dim):
        return self.dimension == dimensionify(dim)

    # for munits support
    def _plot_get_value_for_plot(self, q_unit):  
        q_unit = quantify(q_unit)
        if not self.dimension == q_unit.dimension:
            raise DimensionError(self.dimension, q_unit.dimension)
        return self/q_unit
    
    # for munits support
    def _plot_extract_q_for_axe(self):
        favunit = self.favunit
        if isinstance(favunit, Quantity):
            ratio_favunit = make_quantity(self/favunit)
            dim_SI = ratio_favunit.dimension
            if dim_SI == Dimension(None):
                return favunit
            else:
                return make_quantity(favunit * ratio_favunit._SI_unitary_quantity, 
                                     symbol=str(favunit.symbol) + "*" + ratio_favunit._SI_unitary_quantity.dimension.str_SI_unit())
        else:
            return self._SI_unitary_quantity

    def __getattr__(self, item):
        if item.startswith('__array_'):
            #warnings.warn("The unit of the quantity is stripped.")
            if isinstance(self.value, np.ndarray):
                return getattr(self.value, item)
            else:
                # If an `__array_` attributes is requested but the magnitude is not an ndarray,
                # we convert the magnitude to a numpy ndarray.
                self.value = np.array(self.value)
                return getattr(self.value, item)
        try:
            res = getattr(self.value, item)
            return res
        except AttributeError as ex:
            raise AttributeError("Neither Quantity object nor its value ({}) "
                                 "has attribute '{}'".format(self.value, item))
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
    # TODO : make this a static function ?
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        https://numpy.org/doc/1.18/user/basics.dispatch.html?highlight=array%20interface
        
        The __array_ufunc__ receives:
         - ufunc, a function like numpy.multiply
         - method, a string, differentiating between numpy.multiply(...) and variants like numpy.multiply.outer, numpy.multiply.accumulate, and so on. For the common case, numpy.multiply(...), method == '__call__'.
         - inputs, which could be a mixture of different types
         - kwargs, keyword arguments passed to the function

        This doesn't need __getattr__ nor __array__

        """
        #print(args)
        ufunc_name = ufunc.__name__
        left = quantify(args[0])
        
        if not method == "__call__":
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
        
        if ufunc_name in same_dim_out_2:
            other = quantify(args[1])
            if not left.dimension == other.dimension:
                raise DimensionError(left.dimension, other.dimenion)
            res = ufunc.__call__(left.value,other.value)    
            return Quantity(res, left.dimension)
        elif ufunc_name in skip_2:
            other = quantify(args[1])
            res = ufunc.__call__(left.value, other.value)    
            if ufunc_name == "multiply":
                return Quantity(res, left.dimension * other.dimension)
            elif ufunc_name == 'divide' or ufunc_name == "true_divide":
                return Quantity(res, left.dimension / other.dimension).rm_dim_if_dimless()
        elif ufunc_name in no_dim_1:
            if not left.dimension == Dimension(None):
                raise DimensionError(left.dimension, Dimension(None))
            res = ufunc.__call__(left.value)
            return Quantity(res, Dimension(None))
        elif ufunc_name in angle_1:
            if not left.is_dimensionless_ext():
                raise DimensionError(left.dimension, Dimension(None), binary=True)
            res = ufunc.__call__(left.value)
            return Quantity(res, Dimension(None)).rm_dim_if_dimless()
        elif ufunc_name in same_out:
            res = ufunc.__call__(left.value)
            return Quantity(res, left.dimension).rm_dim_if_dimless()
        elif ufunc_name in special_dict:
            if ufunc_name == "sqrt":
                res = ufunc.__call__(left.value)
                return Quantity(res, left.dimension**(1/2))
            elif ufunc_name == "power":
                power_num = args[1]
                if not (isinstance(power_num,int) or isinstance(power_num,float)):
                    raise TypeError(("Power must be a number, "
                            "not {}").format(type(power_num)))
                res = ufunc.__call__(left.value, power_num)
                return Quantity(res, 
                        left.dimension ** power_num,
                        symbol = left.symbol ** power_num).rm_dim_if_dimless()
            elif ufunc_name == "reciprocal":
                res = ufunc.__call__(left.value)
                return Quantity(res, 1/left.dimension)
            elif ufunc_name == "square":
                res = ufunc.__call__(left.value)
                return Quantity(res, left.dimension**2)
            else:
                raise ValueError
        elif ufunc_name in same_dim_in_2_nodim_out:
            other = quantify(args[1])
            if not left.dimension == other.dimension:
                raise DimensionError(left.dimension, other.dimension)
            res = ufunc.__call__(left.value, other.value)    
            return res
        elif ufunc_name in inv_angle_1:
            if not left.dimension == Dimension(None):
                raise DimensionError(left.dimension, Dimension(None))
            res = ufunc.__call__(left.value)
            return res
        elif ufunc_name in same_dim_in_1_nodim_out:
            res = ufunc.__call__(left.value)
            return res
        else:
            raise ValueError


# Override functions - used with __array_function__
def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator            

@implements(np.sum)
def np_sum(q):
    return Quantity(np.sum(q.value), q.dimension, favunit=q.favunit)

@implements(np.mean)
def np_mean(q):
    return Quantity(np.mean(q.value), q.dimension, favunit=q.favunit)

@implements(np.std)
def np_std(q):
    return Quantity(np.std(q.value), q.dimension)

@implements(np.average)
def np_average(q):
    return Quantity(np.average(q.value), q.dimension, favunit=q.favunit)

@implements(np.median)
def np_average(q):
    return Quantity(np.median(q.value), q.dimension, favunit=q.favunit)

@implements(np.var)
def np_var(q):
    return Quantity(np.var(q.value), q.dimension)

@implements(np.trapz)
def np_trapz(q, **kwargs):
    if not isinstance(q.value,np.ndarray):
            raise TypeError("Quantity value must be array-like to integrate.")
    return Quantity(np.trapz(q.value, **kwargs),
                       q.dimension,
                       favunit = q.favunit)


    


# 2 in : same dimension ---> out : same dim as in
same_dim_out_2 = ["add", "subtract", "hypot", "maximum", "minimum", "fmax", "fmin"]
# 2 in : same dim ---> out : not a quantity
same_dim_in_2_nodim_out = ["greater", "greater_equal", "less", "less_equal", "not_equal", "equal"] # , "logical_and", "logical_or", "logical_xor", "logical_not"]
same_dim_in_1_nodim_out = ["sign"]
# 2 in : any ---> out : depends
skip_2 = ["multiply", "divide", "true_divide"]
# 1 in : any ---> out : depends
special_dict = ["sqrt", "power", "reciprocal", "square"]
# 1 in : no dim ---> out : no dim
no_dim_1 = ["exp", "log"]
# 2 in : no dim ---> out : no dim
no_dim_2 = ["logaddexp", "logaddexp2", ]
# 1 in : dimless or angle ---> out : dimless
angle_1 = ["cos", "sin", "tan", 
          "cosh", "sinh", "tanh"]
# 1 in : any --> out : same
same_out = ["ceil", "conjugate", "floor", "rint", "trunc", "fabs"]
# 1 in : dimless -> out : dimless
inv_angle_1 = ["arcsin", "arccos", "arctan",
              "arcsinh", "arccosh", "arctanh"]
# dimless -> dimless
deg_rad = ["deg2rad", "rad2deg"]

ufunc_2_args = same_dim_out_2 + skip_2 + no_dim_2


def quantify(x):
    if isinstance(x, Quantity):
        return x.__copy__()
    else:
        return Quantity(x, Dimension(None))

def dimensionify(x):
    if isinstance(x, Dimension):
        return x
    elif isinstance(x, Quantity):
        return x.dimension
    elif np.isscalar(x) and not type(x) == str:
        return Dimension(None)
    elif isinstance(x, np.ndarray):
        return Dimension(None)
    else:
        return Dimension(x)


def make_quantity(x, symbol="UndefinedSymbol", favunit=None):
    if isinstance(x, Quantity):
        q = x.__copy__()
        q.symbol = symbol
        if favunit is None:
            if not x.favunit is None:
                q.favunit = x.favunit
        else:
            q.favunit = favunit
        return q
    else:
        return Quantity(x, Dimension(None), symbol=symbol, favunit=favunit)




class QuantityIterator(object):

    def __init__(self, q):
        if not isinstance(q, Quantity):
            raise TypeError("QuantityIterator: must be Quantity object.")
        self.value = q.value
        self.dimension = q.dimension
        self.favunit = q.favunit
        self.symbol = q.symbol
        if isinstance(q.value, np.ndarray):
            self.length = q.value.size
        else:
            self.length = 1
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.length:
            raise StopIteration
        else:
            if isinstance(self.value, np.ndarray):
                q_out = Quantity(self.value[self.count],
                             self.dimension,
                             favunit=self.favunit,
                             symbol=self.symbol)
            else:
                q_out = Quantity(self.value,
                             self.dimension,
                             favunit=self.favunit,
                                symbol=self.symbol)
        self.count += 1

        return q_out


def main():
    pass


if __name__ == "__main__":
    main()
