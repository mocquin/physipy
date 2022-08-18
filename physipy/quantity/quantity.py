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
 - [ ] : improve Inration of eq, ne (ex : assertNotEqual when dealing with arrays)
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

import math
import numbers as nb
import numpy as np
import sympy as sp

import warnings

from .dimension import Dimension, DimensionError, SI_UNIT_SYMBOL, DIMENSIONLESS

# # Constantes
UNIT_PREFIX= " "
DISPLAY_DIGITS = 2
EXP_THRESHOLD = 2
UNIT_SUFFIX = ""
LATEX_VALUE_UNIT_SEPARATOR = "\,"#" \cdot "
#DEFAULT_SYMBOL = sp.Symbol("UndefinedSymbol")
DEFAULT_SYMBOL = "UndefinedSymbol"
#SCIENTIFIC = '%.' + str(DISPLAY_DIGITS) + 'E' # (syntaxe : "%.2f" % mon_nombre
#CLASSIC =  '%.' + str(DISPLAY_DIGITS) + 'f'

HANDLED_FUNCTIONS = {}


class Quantity(object):
    """Quantity class : """
    __quantity_priority__ = 1000
    DIGITS = DISPLAY_DIGITS
    EXP_THRESH = EXP_THRESHOLD
    LATEX_SEP = LATEX_VALUE_UNIT_SEPARATOR
    __array_priority__ = 100
    
    def __init__(self, value, dimension, symbol=DEFAULT_SYMBOL, favunit=None):
        self.value = value
        self.dimension = dimension
        self.symbol = symbol
        self.favunit = favunit

    @property
    def size(self):
        if isinstance(self.value, np.ndarray):
            return self.value.size
        else:
            return 1
        
    #@property
    #def symbol(self):
    #    return self._symbol
    
    #@symbol.setter
    #def symbol(self, value):
    #    if isinstance(value,sp.Expr):                  
    #        self._symbol = value
    #    elif isinstance(value,str):                       
    #        self._symbol = sp.Symbol(value)
    #    else: 
    #        raise TypeError(("Symbol of Quantity must be a string "
    #                         "or a sympy-symbol, "
    #                         "not {}").format(type(value)))
        
    @property
    def favunit(self):
        return self._favunit
    
    @favunit.setter
    def favunit(self, value):
        if isinstance(value,Quantity) or value == None:
            self._favunit = value
        elif np.isscalar(value):
            self._favunit = None
        else:
            raise TypeError(("Favorite unit of Quantity must be a Quantity "
                             "or None, not {}").format(type(value)))
        

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        if isinstance(value, (list, tuple)):
            self._value = np.array(value)
        else:
            self._value = value


    def __add__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension:                                                               
            raise DimensionError(self.dimension, y.dimension)                          
        #return Quantity(self.value + y.value,
        #                self.dimension)    
        return type(self)(self.value + y.value,
                        self.dimension)

    def __radd__(self, x): return self + x

    def __sub__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension: 
            raise DimensionError(self.dimension, y.dimension)
        return type(self)(self.value - y.value,
                        self.dimension)

    def __rsub__(self, x): return quantify(x) - self

    def __mul__(self,y):
        # TODO make a decorator "try_raw_then_quantify_if_fail"
        try:
            return type(self)(self.value * y.value, 
                self.dimension * y.dimension, 
                symbol=self.symbol + "*" + y.symbol,
                 ).rm_dim_if_dimless() 
        except:
            y = quantify(y)
            return type(self)(self.value * y.value, 
                self.dimension * y.dimension, 
                symbol=self.symbol + "*" + y.symbol,
                 ).rm_dim_if_dimless() 
    
    __rmul__ = __mul__
    
    def __matmul__(self, y):
        y = quantify(y)
        return type(self)(self.value @ y.value,
                        self.dimension * y.dimension, 
                        #symbol = self.symbol * y.symbol
                         ).rm_dim_if_dimless() 

    def __truediv__(self, y):
        y = quantify(y)
        return type(self)(self.value / y.value,
                        self.dimension / y.dimension,
                        symbol = self.symbol + "/" + y.symbol,
                         ).rm_dim_if_dimless()

    def __rtruediv__(self, x): return quantify(x) / self

    def __floordiv__(self, y):
        """
        Any returned quantity should be dimensionless, but leaving the 
        Quantity().remove() because more intuitive
        """
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return type(self)(self.value // y.value,
                       self.dimension).rm_dim_if_dimless()
    
    def __rfloordiv__(self, x):
        x = quantify(x)
        if not self.dimension == x.dimension:
            raise DimensionError(self.dimension, x.dimension)
        return type(self)(x.value // self.value,
                       self.dimension).rm_dim_if_dimless()    

    def __mod__(self,y):
        """
        There is no rm_dim_if_dimless() because a 
        modulo operation would not change the dimension.
        
        """
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return type(self)(self.value % y.value,
                        self.dimension)#.rm_dim_if_dimless()

    def __pow__(self,power):
        """
        A power must always be a dimensionless scalar. 
        If a = 1*m, we can't do a ** [1,2], because the result would be
        an array of quantity, and can't be a quantity with array-value, 
        since the quantities won't be the same dimension.
        
        """
        #if not np.isscalar(power):#(isinstance(power,int) or isinstance(power,float)):
        #    raise TypeError(("Power must be a number, "
        #                    "not {}").format(type(power)))
        return type(self)(self.value ** power, 
                        self.dimension ** power,
                        symbol = self.symbol + '**' +str(power),
                       ).rm_dim_if_dimless()

    def __neg__(self): return Quantity(-self.value, self.dimension, favunit=self.favunit)
    
    def __pos__(self): return self
    
    def __len__(self): return len(self.value)
    
    def __bool__(self): return bool(self.value)

    # min and max uses the iterator
    #def __min__(self):
    #    return Quantity(min(self.value),
    #                    self.dimension,
    #                    favunit=self.favunit)

    #def __max__(self):
    #    return Quantity(max(self.value),
    #                    self.dimension,
    #                    favunit=self.favunit)

    def __eq__(self,y):
        try:
            y = quantify(y)
            if self.dimension == y.dimension:
                return self.value == y.value # comparing arrays returns array of bool
            else:
                return False
        except Exception as e:
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
        return type(self)(abs(self.value),
                        self.dimension,
                        favunit = self.favunit)


    def __complex__(self):
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, DIMENSIONLESS, binary=False)
        return complex(self.value)


    def __int__(self):
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, DIMENSIONLESS, binary=False)
        return int(self.value)


    def __float__(self):
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, DIMENSIONLESS, binary=False)
        return float(self.value)


    def __round__(self, i=None):
        return type(self)(round(self.value, i), 
                       self.dimension,
                       favunit = self.favunit)


    def __copy__(self):
        return type(self)(self.value, self.dimension, favunit=self.favunit, symbol=self.symbol)
    
    def copy(self):
        return self.__copy__()
    
    def __repr__(self):
        if str(self.symbol) != "UndefinedSymbol":
            sym = ", symbol="+str(self.symbol)
        else:
            sym = ""
        return f'<{self.__class__.__name__} : ' + str(self.value) + " " + str(self.dimension.str_SI_unit()) + sym + ">"        

    def __str__(self):
        complement_value_for_repr = self._compute_complement_value() 
        if not complement_value_for_repr == "":
            return str(self._compute_value()) + UNIT_PREFIX + complement_value_for_repr + UNIT_SUFFIX
        else: 
            return str(self._compute_value()) + UNIT_SUFFIX
       
    def __hash__(self):
        return hash(str(self.value)+str(self.dimension))
 
        
    def __ceil__(self):
        """
        To handle math.ceil
        """
        return type(self)(math.ceil(self.value), self.dimension)
    
    def __floor__(self):
        """
        To handle math.floor
        """
        return type(self)(math.floor(self.value), self.dimension)
    
    def __trunc__(self):
        return type(self)(math.trunc(self.value), self.dimension)
    
    #@property
    #def latex(self):
    #    return self._repr_latex_()

    #@property
    #def html(self):
    #    return self._repr_html_()

    #def _repr_pretty_(self, p, cycle):
    #    """Markdown hook for ipython repr.
    #    See https://ipython.readthedocs.io/en/stable/config/integrating.html"""
    #    print("repr_pretty")
    #    return p.text(self._repr_latex_())
    
    def plot(self, kind="y", other=None, ax=None):
        from physipy.quantity.plot import plotting_context
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        with plotting_context():
            if kind =="y" and other is None:
                ax.plot(self)
            elif kind =="x" and other is not None:
                ax.plot(self, other)
            else:
                raise ValueError("kind must be y of x with other")
    
    def _repr_latex_(self):
        """Markdown hook for ipython repr in latex.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html"""

        # create a copy
        q = self.__copy__()
        # to set a favunit for display purpose
        # only change the favunit if not already defined
        if q.favunit == None:
            q.favunit = self._pick_smart_favunit()
        formatted_value = q._format_value()
        complemented = q._compute_complement_value()
        if complemented != "":
            # this line simplifies 'K*s/K' when a = 1*s and c = a.to(K)
            complement_value_str = sp.printing.latex(sp.parsing.sympy_parser.parse_expr(complemented))
        else:
            complement_value_str = ""
        # if self.value is an array, only wrap the complement in latex
        if isinstance(self.value, np.ndarray):
            return formatted_value + "$" + self.LATEX_SEP + complement_value_str + "$"
        # if self.value is a scalar, use sympy to parse expression
        value_str = sp.printing.latex(sp.parsing.sympy_parser.parse_expr(formatted_value))
        return "$" + value_str + self.LATEX_SEP + complement_value_str + "$"
    
    
    def _pick_smart_favunit(self, array_to_scal=np.mean):
        """Method to pick the best favunit among the units dict.
        A smart favunit always have the same dimension as self.
        The 'best' favunit is the one minimizing the difference with self.
        In case self.value is an array, array_to_scal is 
        used to convert the array to a single value.
        """
        from .units import units
        from .utils import asqarray
        same_dim_unit_list = [value for value in units.values() if self.dimension == value.dimension]
        # if no unit with same dim already exists
        if len(same_dim_unit_list) == 0:
            return None
        same_dim_unit_arr = asqarray(same_dim_unit_list)
        self_val = self if not isinstance(self.value, np.ndarray) else array_to_scal(self)
        best_ixd = np.abs(same_dim_unit_arr - np.abs(self_val)).argmin()
        best_favunit = same_dim_unit_list[best_ixd]
        return best_favunit
    

    def _format_value(self):
        """Used to format the value on repr as a str.
        If the value is > to 10**self.EXP_THRESH, it is displayed with scientific notation.
        Else floating point notation is used.
        """
        value = self._compute_value()
        if not np.isscalar(value):
            return str(value)
        else:
            if abs(value) >= 10**self.EXP_THRESH or abs(value) < 10**(-self.EXP_THRESH):
                return ("{:." + str(self.DIGITS) + "e}").format(value)
            else:
                return ("{:." + str(self.DIGITS) + "f}").format(value)

    
    #def _repr_markdown_(self):
    #    """Markdown hook for ipython repr in markdown.
    #    this seems to take precedence over _repr_latex_"""
    #    return self.__repr__()

    #def _repr_html(self):
    #    return self._repr_latex_()

    #def __format_raw__(self, format_spec):
    #    return format(self.value, format_spec) + " " + str(self.dimension.str_SI_unit())

    def __format__(self, format_spec):
        """This method is used when using format or f-string. 
        The format is applied to the numerical value part only."""
        complement_value_for_repr = self._compute_complement_value()
        prefix = UNIT_PREFIX
        if "~" in format_spec:
            format_spec = format_spec.replace("~", "")
            prefix = ""
        if not complement_value_for_repr == "":
            return format(self._compute_value(), format_spec) + prefix + complement_value_for_repr + UNIT_SUFFIX
        else: 
            return format(self._compute_value(), format_spec) + prefix
    
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
            if dim_SI == DIMENSIONLESS:
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
        return type(self)(1, self.dimension, symbol=self.dimension.str_SI_unit())

    
    def __getitem__(self, idx):
        """
        Having this defined here makes iter(m) not raising an exception
        so np.iterable considers m as iterable, which is not.
        Maybe use monkey patching ?
        Solution was to define __iter__ since iter first checks that
        x.__iter__ doesn't raise a TypeError
        """
        return type(self)(self.value[idx],
                            self.dimension,
                            favunit=self.favunit)

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
        """
        Having __iter__ makes isinstance(x, collections.abc.Iterable) return True
        just because x has attr "__iter__".
        Was moved to getattr, but come back for compatibility with iter(m)
        and np.iterable.
        """
        iter(self.value)
        if isinstance(self.value,np.ndarray):
            return QuantityIterator(self)
        else:
            return iter(self.value)
        
    @property
    def flat(self):
        # pint implementation
        #for v in self.value.flat:
        #    yield Quantity(v, self.dimension)
        
        # astropy
        return FlatQuantityIterator(self)
    
    def flatten(self):
        return type(self)(self.value.flatten(), self.dimension, favunit=self.favunit)
    
    def tolist(self):
        return [type(self)(i, self.dimension) for i in self.value]
    
    @property
    def real(self):
        return type(self)(self.value.real, self.dimension)
    
    @property
    def imag(self):
        return type(self)(self.value.imag, self.dimension)
    
    @property
    def T(self):
        return type(self)(self.value.T, self.dimension)
    
    
    def std(self, *args, **kwargs):
        return np.std(self, *args, **kwargs)
    
    def inverse(self):
        """is this method usefull ?"""
        return type(self)(1/self.value, 1/self.dimension)

    def sum(self, **kwargs): return np.sum(self, **kwargs)
    
    def mean(self, **kwargs): return np.mean(self, **kwargs)
    
    def integrate(self, *args, **kwargs): return np.trapz(self, *args, **kwargs)
    
    def is_dimensionless(self):
        return self.dimension == DIMENSIONLESS

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
    
    def set_favunit(self, fav):
        """
        To be used as one-line declaration : (np.linspace(3, 10)*mum).set_favunit(mum)
        """
        self.favunit = fav
        return self

    def set_symbol(self, symbol):
        """
        To be used as one-line declaration for a favunit : my_period=(10*s).set_symbol("period")
        """
        self.symbol = symbol
        return self
    
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

    def is_nan(self):
        "For use with pandas extension"
        return np.isnan(self.value)

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
    def _plot_extract_q_for_axe(self, units_list):
        if self.favunit is None:
            favs = [unit for unit in units_list if self._SI_unitary_quantity==unit]
            if len(favs)>=1:
                favunit = favs[0]
            else:
                favunit = self.favunit
        else:
            favunit = self.favunit        
        if isinstance(favunit, Quantity):
            ratio_favunit = make_quantity(self/favunit)
            dim_SI = ratio_favunit.dimension
            if dim_SI == DIMENSIONLESS:
                return favunit
            else:
                return make_quantity(favunit * ratio_favunit._SI_unitary_quantity, 
                                     symbol=str(favunit.symbol) + "*" + ratio_favunit._SI_unitary_quantity.dimension.str_SI_unit())
        else:
            return self._SI_unitary_quantity

    def __getattr__(self, item):
        """
        Called when an attribute lookup has not found the attribute 
        in the usual places (i.e. it is not an instance attribute
        nor is it found in the class tree for self). name is the 
        attribute name. This method should return the (computed) 
        attribute value or raise an AttributeError exception.
        Note that if the attribute is found through the normal mechanism, 
        __getattr__() is not called.
        """
        #if item == '__iter__':
        #   if isinstance(self.value,np.ndarray):
        #        return QuantityIterator(self)
        #    else:
        #        return iter(self.value)
        if item.startswith('__array_'):
            warnings.warn(f"The unit of the quantity is stripped for {item}")
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
    
    #def to_numpy(self):
    #    """
    #    Needed for plt.hist(np.arange(10)*m).
    #    """
    #    return np.asarray(self.value)
    
    def reshape(self, *args, **kwargs):
        return type(self)(self.value.reshape(*args, **kwargs), self.dimension)
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        #if not all(issubclass(t, self.__class__) for t in types):
        #    return NotImplemented
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
        if method == "__call__":
            return self._ufunc_call(ufunc, method, *args, **kwargs)
        elif method == "reduce":
            return self._ufunc_reduce(ufunc, method, *args, **kwargs)
        elif method == "accumulate":
            return self._ufunc_accumulate(ufunc, method, *args, **kwargs)
        else:
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")

            
    def _ufunc_accumulate(self, ufunc, method, *args, **kwargs):
        ufunc_name = ufunc.__name__
        left = args[0]
        # hypot doesn't have a reduce
        if ufunc_name in ["add"]:
            res = ufunc.accumulate(left.value, **kwargs)    
            return type(self)(res, left.dimension)
        else:
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
            
    def _ufunc_reduce(self, ufunc, method, *args, **kwargs):
        """
        The method == "reduce" part of __array_ufunc__ interface.
        """
        if not method == "reduce":
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
        ufunc_name = ufunc.__name__
        left = args[0] # removed quantify...
                
        # hypot doesn't have a reduce
        if ufunc_name in same_dim_out_2 and ufunc_name != "hypot":
            res = ufunc.reduce(left.value, **kwargs)
            return type(self)(res, left.dimension)
        # only multiply seems to be possible
        elif ufunc_name in skip_2:
            res = ufunc.reduce(left.value, **kwargs)    
            if ufunc_name == "multiply" or ufunc_name == "matmul":
                return type(self)(res, left.dimension ** len(left.value))
            else:
                raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
        # return booleans : 
        elif ufunc_name in same_dim_in_2_nodim_out:
            res =  ufunc.reduce(left.value, **kwargs)    
            return res
        # ValueError: reduce only supported for binary functions
        elif ufunc_name in unary_ufuncs:
            raise ValueError("reduce only supported for binary functions.")
        else:
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
        
    def _ufunc_call(self, ufunc, method, *args, **kwargs):
        """
        The method == "__call__" part of __array_ufunc__ interface.
        """
        if not method == "__call__":
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
        ufunc_name = ufunc.__name__
        left = quantify(args[0])
        
        if ufunc_name in same_dim_out_2:
            other = quantify(args[1])
            if not left.dimension == other.dimension:
                raise DimensionError(left.dimension, other.dimension)
            res = ufunc.__call__(left.value,other.value)    
            return type(self)(res, left.dimension)
        elif ufunc_name in skip_2:
            other = quantify(args[1])
            res = ufunc.__call__(left.value, other.value)    
            if ufunc_name == "multiply" or ufunc_name == "matmul":
                return type(self)(res, left.dimension * other.dimension)
            elif ufunc_name == 'divide' or ufunc_name == "true_divide":
                return type(self)(res, left.dimension / other.dimension).rm_dim_if_dimless()
            elif ufunc_name == "copysign" or ufunc_name == "nextafter":
                return type(self)(res, left.dimension)
        elif ufunc_name in no_dim_1:
            if not left.dimension == DIMENSIONLESS:
                raise DimensionError(left.dimension, DIMENSIONLESS)
            res = ufunc.__call__(left.value)
            return type(self)(res, DIMENSIONLESS)
        elif ufunc_name in angle_1:
            if not left.is_dimensionless_ext():
                raise DimensionError(left.dimension, DIMENSIONLESS, binary=True)
            res = ufunc.__call__(left.value)
            return type(self)(res, DIMENSIONLESS).rm_dim_if_dimless()
        elif ufunc_name in same_out:
            res = ufunc.__call__(left.value)
            return type(self)(res, left.dimension).rm_dim_if_dimless()
        elif ufunc_name in special_dict:
            if ufunc_name == "sqrt":
                res = ufunc.__call__(left.value)
                return type(self)(res, left.dimension**(1/2))
            elif ufunc_name == "power":
                power_num = args[1]
                if not (isinstance(power_num,int) or isinstance(power_num,float)):
                    raise TypeError(("Power must be a number, "
                            "not {}").format(type(power_num)))
                res = ufunc.__call__(left.value, power_num)
                return type(self)(res, 
                        left.dimension ** power_num,
                        symbol = left.symbol ** power_num).rm_dim_if_dimless()
            elif ufunc_name == "reciprocal":
                res = ufunc.__call__(left.value)
                return type(self)(res, 1/left.dimension)
            elif ufunc_name == "square":
                res = ufunc.__call__(left.value)
                return type(self)(res, left.dimension**2)
            elif ufunc_name == "cbrt":
                res = ufunc.__call__(left.value)
                return type(self)(res, left.dimension**(1/3))
            elif ufunc_name == "modf":
                res = ufunc.__call__(left.value)
                frac, integ = res
                return (type(self)(frac, left.dimension),
                       type(self)(integ, left.dimension))
            elif ufunc_name == "arctan2":
                # both x and y should have same dim such that the ratio is dimless
                other = quantify(args[1])
                if not left.dimension == other.dimension:
                    raise DimensionError(left.dimension, other.dimension)
                # use the value so that the 0-comparison works
                res = ufunc.__call__(left.value, other.value, **kwargs)
                return res
            else:
                raise ValueError
        elif ufunc_name in same_dim_in_2_nodim_out:
            other = quantify(args[1])
            if not left.dimension == other.dimension:
                raise DimensionError(left.dimension, other.dimension)
            res = ufunc.__call__(left.value, other.value)    
            return res
        elif ufunc_name in inv_angle_1:
            if not left.dimension == DIMENSIONLESS:
                raise DimensionError(left.dimension, DIMENSIONLESS)
            res = ufunc.__call__(left.value)
            return res
        #elif ufunc_name in inv_angle_2:
        #    other = quantify(args[1])
        #    if not (left.dimension == DIMENSIONLESS and other.dimension == DIMENSIONLESS):
        #        raise DimensionError(left.dimension, DIMENSIONLESS)
        #    res = ufunc.__call__(left.value, other.value)
        #    return res
        elif ufunc_name in same_dim_in_1_nodim_out:
            res = ufunc.__call__(left.value)
            return res
        elif ufunc_name in no_dim_2:
            other = quantify(args[1])
            if not (left.dimension == DIMENSIONLESS and other.dimension == DIMENSIONLESS):
                raise DimensionError(left.dimension, DIMENSIONLESS)
            res = ufunc.__call__(left.value, other.value)
            return res
        else:
            raise ValueError("ufunc not implemented ?: ", str(ufunc))

            
    def squeeze(self, *args, **kwargs):
        """
        Helper function to wrap numpy's squeeze.
        """
        return type(self)(self.value.squeeze(*args, **kwargs), self.dimension)

# Numpy functions            
# Override functions - used with __array_function__
def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator            

@implements(np.asanyarray)
def np_asanyarray(a):
    return Quantity(np.asanyarray(a.value), a.dimension)



@implements(np.amax)
def np_amax(q): return Quantity(np.amax(q.value), q.dimension, favunit=q.favunit)

@implements(np.amin)
def np_amin(q): return Quantity(np.amin(q.value), q.dimension, favunit=q.favunit)

@implements(np.append)
def np_append(arr, values, **kwargs):
    values = quantify(values)
    if not arr.dimension == values.dimension:
        raise DimensionError(arr.dimension, values.dimension)
    return Quantity(np.append(arr.value, values.value, **kwargs), arr.dimension)

@implements(np.argmax)
def np_argmax(a, **kwargs):
    return Quantity(np.argmax(a.value, **kwargs), a.dimension)


@implements(np.argsort)
def np_argsort(a, **kwargs):
    return np.argsort(a.value, **kwargs)

@implements(np.sort)
def np_sort(a, **kwargs):
    return Quantity(np.sort(a.value, **kwargs), a.dimension)


@implements(np.argmin)
def np_argmin(a, **kwargs):
    return Quantity(np.argmin(a.value, **kwargs), a.dimension)


@implements(np.around)
def np_around(a, **kwargs):
    return Quantity(np.around(a.value, **kwargs), a.dimension)


@implements(np.atleast_1d)
def np_atleast_1d(*arys):
    res = [Quantity(np.atleast_1d(arr.value), arr.dimension) for arr in arys] 
    return res if len(res)>1 else res[0]


@implements(np.atleast_2d)
def np_atleast_2d(*arys):
    res = [Quantity(np.atleast_2d(arr.value), arr.dimension) for arr in arys] 
    return res if len(res)>1 else res[0]


@implements(np.atleast_3d)
def np_atleast_3d(*arys):
    res = [Quantity(np.atleast_3d(arr.value), arr.dimension) for arr in arys] 
    return res if len(res)>1 else res[0]


@implements(np.average)
def np_average(q): return Quantity(np.average(q.value), q.dimension, favunit=q.favunit)

# np.block : todo

@implements(np.broadcast_to)
def np_broadcast_to(array, *args, **kwargs):
    return Quantity(np.broadcast_to(array.value, *args, **kwargs), array.dimension)


@implements(np.broadcast_arrays)
def np_broadcast_arrays(*args, **kwargs):
    qargs = [quantify(a) for a in args]
    # get arrays values
    arrs = [qarg.value for qarg in qargs]
    # get broadcasted arrays
    res = np.broadcast_arrays(*arrs, **kwargs)
    return [Quantity(r, q.dimension) for r, q in zip(res, qargs)]

@implements(np.linalg.lstsq)
def np_linalg_lstsq(a, b, **kwargs):
    a = quantify(a)
    b = quantify(b)
    sol = np.linalg.lstsq(a.value, b.value, **kwargs)
    return Quantity(sol, b.dimension/a.dimension)

@implements(np.linalg.inv)
def np_inv(a):
    return Quantity(np.linalg.inv(a.value), 1/a.dimension)


@implements(np.flip)
def np_flip(m, axis=None):
    return Quantity(np.flip(m.value, axis=axis), 
                    m.dimension,
                    symbol=m.symbol, 
                    favunit=m.favunit)


@implements(np.fliplr)
def np_fliplr(m):
    return Quantity(np.fliplr(m.value),
                    m.dimension, 
                    symbol=m.symbol, 
                    favunit=m.favunit)


@implements(np.flipud)
def np_flipud(m):
    return Quantity(np.flipud(m.value),
                    m.dimension, 
                    symbol=m.symbol,
                    favunit=m.favunit)


@implements(np.random.normal)
def np_random_normal(loc=0.0, scale=1.0, **kwargs):
    loc = quantify(loc)
    scale = quantify(scale)
    if not loc.dimension == scale.dimension:
        raise DimensionError(loc.dimension, scale.dimension)
    return Quantity(np.random.normal(loc=loc.value, scale=scale.value,
                                **kwargs))

@implements(np.may_share_memory)
def np_may_share_memory(a, b, **kwargs):
    return np.may_share_memory(a.value, b.value, **kwargs)


@implements(np.polyfit)
def np_polyfit(x, y, deg, *args, **kwargs):
    x = quantify(x)
    y = quantify(y)
    p = np.polyfit(x.value, y.value, deg, *args, **kwargs)
    qp = tuple(Quantity(coef, y.dimension / x.dimension**(deg-i) ) for i, coef in enumerate(p))
    return qp
       
@implements(np.polyval)
def np_polyval(p, x):
    p_values = tuple(quantify(coef).value for coef in p)
    x = quantify(x)
    res = np.polyval(p_values, x.value)
    return Quantity(res, p[-1].dimension)
 


@implements(np.clip)
def np_clip(a, a_min, a_max, *args, **kwargs):
    a_min = quantify(a_min)
    a_max = quantify(a_max)
    if a.dimension != a_min.dimension:
        raise DimensionError(a.dimension, a_min.dimension)
    if a.dimension != a_max.dimension:
        raise DimensionError(a.dimension, a_max.dimension)
    return Quantity(np.clip(a.value, a_min.value,
                            a_max.value, *args, **kwargs), a.dimension)

@implements(np.copyto)
def np_copyto(dst, src, **kwargs):
    dst = quantify(dst)
    src = quantify(src)
    if dst.dimension != src.dimension:
        raise DimensionError(dst.dimension, src.dimension)
    return np.copyto(dst.value, src.value, **kwargs)


@implements(np.column_stack)
def np_column_stack(tup):
    dim = tup[0].dimension
    for arr in tup:
        if arr.dimension != dim:
            raise DimensionError(arr.dimension, dim)
    return Quantity(np.column_stack(tuple(arr.value for arr in tup)), dim)


@implements(np.compress)
def np_compress(condition, a, **kwargs):
    return Quantity(np.compress(condition, a.value, **kwargs), a.dimension)


@implements(np.concatenate)
def np_concatenate(tup, axis=0, out=None):
    dim = tup[0].dimension
    for arr in tup:
        if arr.dimension != dim:
            raise DimensionError(arr.dimension, dim)
    return Quantity(np.concatenate(tuple(arr.value for arr in tup)), dim)


@implements(np.copy)
def np_copy(a, **kwargs):
    return Quantity(np.copy(a.value, **kwargs), a.dimension)


# np.copyto todo
# np.count_nonzero

@implements(np.count_nonzero)
def np_count_nonzero(a, *args, **kwargs):
    return np.count_nonzero(a.value, *args, **kwargs)

@implements(np.cross)
def np_cross(a, b, **kwargs):
    return Quantity(np.cross(a.value, b.value),
                    a.dimension*b.dimension)


# np.cumprod : cant have an array with different dimensions

@implements(np.cumsum)
def np_cumsum(a, **kwargs):
    return Quantity(np.cumsum(a.value, **kwargs), a.dimension)


@implements(np.histogram)
def np_histogram(a, bins=10, range=None, normed=None, weights=None, **kwargs):
    if range is not None:
        range = (quantify(range[0]), quantify(range[1]))
        if not range[0].dimension == range[1].dimension:
            raise DimensionError(range[0].dimension, range[1].dimension)
    hist, bin_edges = np.histogram(a.value, bins=bins, range=range, normed=normed, weights=weights)
    return hist, Quantity(bin_edges, a.dimension)

    
@implements(np.diagonal)
def np_diagonal(a, **kwargs):
    return Quantity(np.diagonal(a.value, **kwargs), a.dimension)


@implements(np.diff)
def np_diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    if prepend != np._NoValue:
        if prepend.dimension != a.dimension:
            raise DimensionError(a.dimension, prepend.dimension)
    if append != np._NoValue:
        if append.dimension != a.dimension:
            raise DimensionError(a.dimension, append.dimension)
    return Quantity(np.diff(a.value, n=n, axis=axis, prepend=prepend, append=append), a.dimension)

@implements(np.ndim)
def np_ndim(a):
    return np.ndim(a.value)

@implements(np.dot)
def np_dot(a, b, **kwargs):
    a = quantify(a)
    b = quantify(b)
    return Quantity(np.dot(a.value, b.value), a.dimension * b.dimension)


@implements(np.cov)
def np_cov(m, y=None, *args, **kwargs):
    m = quantify(m)
    if y is not None:
        y = quantify(y)
        raw = np.cov(m.value, y.value, *args, **kwargs)
        return Quantity(raw, m.dimension*y.dimension)
    raw = np.cov(m.value, y, *args, **kwargs)
    return Quantity(raw, m.dimension**2)


@implements(np.percentile)
def np_percentile(a, *args, **kwargs):
    a = quantify(a)
    return Quantity(np.percentile(a.value, *args, **kwargs), a.dimension)


@implements(np.searchsorted)
def np_searchsorted(a, v, *args, **kwargs):
    a = quantify(a)
    v = quantify(v)
    if not a.dimension == v.dimension:
        raise DimensionError(a.dimension, v.dimension)
    return np.searchsorted(a.value, v.value, *args, **kwargs)

@implements(np.stack)
def np_stack(arrays, *args, **kwargs):
    d = quantify(arrays[0]).dimension
    ars = []
    for ar in arrays:
        qar = quantify(ar)
        if qar.dimension ==d:
            ars.append(qar.value)
        else:
            raise DimensionError(d, qar.dimension)
    return Quantity(np.stack(ars, *args, **kwargs), d)

@implements(np.insert)
def np_insert(arr, obj, values, *args, **kwargs):
    arr = quantify(arr)
    values = quantify(values)
    if not arr.dimension == values.dimension:
        raise DimensionError(arr.dimension, values.dimension)
    return Quantity(np.insert(arr.value, obj, values.value, *args, **kwargs), 
                    arr.dimension, favunit=arr.favunit)

@implements(np.dstack)
def np_dstack(tup):
    dim = tup[0].dimension
    for arr in tup:
        if arr.dimension != dim:
            raise DimensionError(arr.dimension, dim)
    return Quantity(np.dstack(tuple(arr.value for arr in tup)), dim)

@implements(np.tile)
def np_tile(A, reps):
    return Quantity(np.tile(A.value, reps), A.dimension)

@implements(np.prod)
def np_prod(a, **kwargs):
    return Quantity(np.prod(a.value), a.dimension**(len(a)))

#@implements(np.ediff1d)
#def np_ediff1d(ary, to_end=None, to_begin=None):
#    if not ary.dimension == to_end.dimension:
#        raise DimensionError(ary.dimension, to_end.dimension)
#    if not to_begin is None:
#        if not ary.dimension == to_begin.dimension:
#             raise DimensionError(ary.dimension, to_begin.dimension)
#    return Quantity(np.ediff1d(ary.value, to_end, to_begin))

@implements(np.may_share_memory)
def np_may_share_memory(a, b, max_work=None):
    if not isinstance(b, Quantity):
        return np.may_share_memory(a.value, b, max_work=max_work)
    if not isinstance(a, Quantity):
        return np.may_share_memory(a, b.value, max_work=max_work)
    return np.may_share_memory(a.value, b.value, max_work=max_work)

@implements(np.sum)
def np_sum(q, **kwargs): 
    return Quantity(np.sum(q.value, **kwargs), q.dimension, favunit=q.favunit)


@implements(np.mean)
def np_mean(q, **kwargs): return Quantity(np.mean(q.value, **kwargs), q.dimension, favunit=q.favunit)


@implements(np.std)
def np_std(q, *args, **kwargs): return Quantity(np.std(q.value, *args, **kwargs), q.dimension, favunit=q.favunit)


@implements(np.median)
def np_median(q): return Quantity(np.median(q.value), q.dimension, favunit=q.favunit)


@implements(np.var)
def np_var(q, *args, **kwargs): return Quantity(np.var(q.value, *args, **kwargs), q.dimension**2)


@implements(np.rollaxis)
def np_rollaxis(a, axis, start=0):
    return Quantity(np.rollaxis(a.value, axis, start=0),
                    a.dimension, 
                    symbol=a.symbol,
                    favunit=a.favunit)

@implements(np.trapz)
def np_trapz(q, x=None, dx=1, **kwargs):
    #if not isinstance(q.value,np.ndarray):
    #        raise TypeError("Quantity value must be array-like to integrate.")
    q = quantify(q)
    if x is None:    
        dx = quantify(dx)
        return Quantity(np.trapz(q.value, x=None, dx=dx.value, **kwargs),
                    q.dimension * dx.dimension,
                    )
    else:
        x = quantify(x)
        return Quantity(np.trapz(q.value, x=x.value, **kwargs),
                    q.dimension * x.dimension,
                    )
    
@implements(np.transpose)
def np_transpose(a, axis=None):
    return Quantity(np.transpose(a.value, axis=axis), 
                    a.dimension,
                    favunit=a.favunit,
                    symbol=a.symbol)

@implements(np.rot90)
def np_rot90(m, k=1, axes=(0,1)):
    return Quantity(np.rot90(m.value, k=k, axes=axes), 
                    m.dimension,
                    favunit=m.favunit,
                    symbol=m.symbol)


@implements(np.lib.stride_tricks.sliding_window_view)
def np_lib_stride_tricks_sliding_window_view(x, *args, **kwargs):
    raw = np.lib.stride_tricks.sliding_window_view(x.value, *args, **kwargs)
    return Quantity(raw, x.dimension, favunit=x.favunit, symbol=x.symbol)

#@implements(np.all)
#def np_all(a, *args, **kwargs):
#    # should dimension also be checked ?
#    return np.all(a.value)


@implements(np.ones_like)
def np_ones_like(a, **kwargs):
    return np.ones_like(a.value, **kwargs)

@implements(np.zeros_like)
def np_zeros_like(a, **kwargs):
    return np.zeros_like(a.value, **kwargs)

@implements(np.zeros)
def np_zeros(shape, dtype=float, order='C', *, like=None):
    like = quantify(like)
    return Quantity(np.zeros(shape, dtype=dtype, order=order), like.dimension)
    

@implements(np.full_like)
def np_full_like(a, fill_value, **kwargs):
    a = quantify(a)
    fill_value = quantify(fill_value)
    return Quantity(np.full_like(a.value, fill_value.value, **kwargs),
                    fill_value.dimension)

@implements(np.empty_like)
def np_empty_like(prototype, **kwargs):
    return np.empty_like(prototype.value, **kwargs)


@implements(np.expand_dims)
def np_expand_ims(a, axis):
    return Quantity(np.expand_dims(a.value, axis), a.dimension)

@implements(np.shape)
def np_shape(a):
    return np.shape(a.value)

#_linspace = decorate_with_various_unit(("A", "A"), "A")(np.linspace)
@implements(np.linspace)
def np_linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start = quantify(start)
    stop = quantify(stop)
    if not start.dimension == stop.dimension:
        raise DimensionError(start.dimension, stop.dimension)
    return Quantity(np.linspace(start.value, stop.value, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis),
                  start.dimension)

@implements(np.corrcoef)
def np_corrcoef(x, y=None, *args, **kwargs):
    x = quantify(x)
    if y is not None:
        y = quantify(y).value
    return np.corrcoef(x.value, y, *args, **kwargs)


@implements(np.take)
def np_take(a, *args, **kwargs):
    return Quantity(np.take(a.value, *args, **kwargs), a.dimension)


@implements(np.squeeze)
def np_squeeze(a, *args):
    return Quantity(np.squeeze(a.value, *args), a.dimension)

@implements(np.repeat)
def np_repeat(a, *args, **kwargs):
    return Quantity(np.repeat(a.value, *args, **kwargs), a.dimension)

@implements(np.roll)
def np_roll(a, *args, **kwargs):
    return Quantity(np.roll(a.value, *args, **kwargs), a.dimension)

@implements(np.meshgrid)
def np_meshgrid(*xi, **kwargs):
    xiq = [quantify(x) for x in xi]
    res = np.meshgrid(*(xi.value for xi in xiq), **kwargs)
    return tuple(Quantity(r, q.dimension) for r, q in zip(res, xiq))


@implements(np.real)
def np_real(a):
    return Quantity(np.real(a.value), a.dimension)

@implements(np.ravel)
def np_ravel(a, *args, **kwargs):
    return Quantity(np.ravel(a.value, *args, **kwargs), a.dimension)
    
@implements(np.reshape)
def np_reshape(a, *args, **kwargs):
    return Quantity(np.reshape(a.value, *args, **kwargs), a.dimension)

@implements(np.interp)
def np_interp(x, xp, fp, left=None, right=None, *args, **kwargs):
    x = quantify(x)
    xp = quantify(xp)
    fp = quantify(fp)
    if not x.dimension == xp.dimension:
        raise DimensionError(x.dimension, xp.dimension)
    if left is not None:
        left = quantify(left)
        if not left.dimension == fp.dimension:
            raise DimensionError(left.dimension, xp.dimension)
        left_v = left.value
    else:
        left_v = left
    if right is not None:
        right = quantify(right)
        if not left.dimension == fp.dimension:
            raise DimensionError(right.dimension, xp.dimension)
        right_v = right.value
    else:
        right_v = right
    
    res = np.interp(x.value, xp.value, fp.value, left_v, right_v, *args, **kwargs)
    return Quantity(res, fp.dimension)

#@implements(np.asarray)
#def np_array(a):
#    print("np_array implm phyispy")
#    return np.asarray(a.value)*m
#
#
#@implements(np.empty)
#def np_empty(shape, dtype=float, order='C'):
#    return np.empty(shape,dtype=float, order=order)
#
@implements(np.full)
def np_full(shape, fill_value, dtype=None, order='C'):
    print("In np_full")
    if dtype is None:
        fill_value = np.asarray(fill_value)
        dtype = fill_value.dtype
    a = np.empty(shape, dtype, order)
    np.copyto(a, fill_value, casting='unsafe')
    return a
    
    
@implements(np.fft.fft)
def np_fft_fft(a, *args, **kwargs):
    """Numpy fft.fft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifft)
def np_fft_ifft(a, *args, **kwargs):
    """Numpy fft.ifft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.fft2)
def np_fft_fft2(a, *args, **kwargs):
    """Numpy fft.fft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifft2)
def np_fft_ifft2(a, *args, **kwargs):
    """Numpy fft.ifft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.fftn)
def np_fft_fftn(a, *args, **kwargs):
    """Numpy fft.fftn wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifftn)
def np_fft_ifftn(a, *args, **kwargs):
    """Numpy fft.ifftn wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.rfft)
def np_fft_rfft(a, *args, **kwargs):
    """Numpy fft.rfft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.rfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.irfft)
def np_fft_irfft(a, *args, **kwargs):
    """Numpy fft.irfft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.irfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.rfft2)
def np_fft_rfft2(a, *args, **kwargs):
    """Numpy fft.rfft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.rfft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.irfft2)
def np_fft_irfft2(a, *args, **kwargs):
    """Numpy fft.irfft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.irfft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.rfftn)
def np_fft_rfftn(a, *args, **kwargs):
    """Numpy fft.ifft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.rfftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.irfftn)
def np_fft_irfftn(a, *args, **kwargs):
    """Numpy fft.irfftn wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.irfftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.hfft)
def np_fft_hfft(a, *args, **kwargs):
    """Numpy fft.httf wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.hfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ihfft)
def np_fft_ihfft(a, *args, **kwargs):
    """Numpy fft.ihfft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ihfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


#@implements(np.fft.fftfreq)
#def np_fft_fftfreq(n, d=1.0):
#    """No need because fftfreq is only a division which is already handled by quantities"""
#    

@implements(np.fft.fftshift)
def np_fft_fftshift(a, *args, **kwargs):
    """Numpy fft.fftshift wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fftshift(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifftshift)
def np_fft_ifftshift(a, *args, **kwargs):
    """Numpy fft.ifftshift wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifftshift(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.convolve)
def np_convolve(a, v, *args, **kwargs):
    a = quantify(a)
    v = quantify(v)
    res = np.convolve(a.value, v.value, **kwargs)
    return Quantity(res, a.dimension * v.dimension)


@implements(np.gradient)
def np_gradient(f, *varargs, **kwargs):
    if len(varargs)>1:
        raise NotImplementedError("High dimension not implemented (but very doable")
    dx = quantify(varargs[0])
    f = quantify(f)
    return Quantity(np.gradient(f.value, dx.value, **kwargs), f.dimension/dx.dimension)

@implements(np.vstack)
def np_vstack(tup):
    dim = tup[0].dimension
    new_tup = []
    for t in tup:
        t = quantify(t)
        if not t.dimension == dim:
            raise DimensionError(dim, t.dimension)
        new_tup.append(t.value)
    return Quantity(np.vstack(new_tup), dim)

@implements(np.hstack)
def np_hstack(tup):
    dim = tup[0].dimension
    new_tup = []
    for t in tup:
        t = quantify(t)
        if not t.dimension == dim:
            raise DimensionError(dim, t.dimension)
        new_tup.append(t.value)
    return Quantity(np.hstack(new_tup), dim)


@implements(np.where)
def np_where(cond, x, y):
    x = quantify(x)
    y = quantify(y)
    if not x.dimension == y.dimension:
        raise DimensionError(x.dimension, y.dimension)
    return Quantity(np.where(cond, x.value, y.value), x.dimension)


# 2 in : same dimension ---> out : same dim as in
same_dim_out_2 = ["add", "subtract", "hypot", "maximum", "minimum", "fmax", "fmin", "remainder", "mod", "fmod"]
# 2 in : same dim ---> out : not a quantity
same_dim_in_2_nodim_out = ["greater", "greater_equal", "less", "less_equal", "not_equal", "equal", "floor_divide"] 
# 1 in : 
same_dim_in_1_nodim_out = ["sign", "isfinite", "isinf", "isnan"]
# 2 in : any ---> out : depends
skip_2 = ["multiply", "divide", "true_divide", "copysign", "nextafter", "matmul"]
# 1 in : any ---> out : depends
special_dict = ["sqrt", "power", "reciprocal", "square", "cbrt", "modf", "arctan2"]
# 1 in : no dim ---> out : no dim
no_dim_1 = ["exp", "log", "exp2", "log2", "log10",
           "expm1", "log1p"]
# 2 in : no dim ---> out : no dim
no_dim_2 = ["logaddexp", "logaddexp2", ]
# 1 in : dimless or angle ---> out : dimless
angle_1 = ["cos", "sin", "tan", 
          "cosh", "sinh", "tanh"]
# 1 in : any --> out : same
same_out = ["ceil", "conjugate", "conj", "floor", "rint", "trunc", "fabs", "negative", "absolute"]
# 1 in : dimless -> out : dimless
inv_angle_1 = ["arcsin", "arccos", "arctan",
              "arcsinh", "arccosh", "arctanh",
              ]
# 1 in : dimless -> dimless
deg_rad = ["deg2rad", "rad2deg"]


not_implemented_yet = ["isreal", "iscomplex", "signbit", "ldexp", "frexp"]
cant_be_implemented = ["logical_and", "logical_or", "logical_xor", "logical_not"]


ufunc_2_args = same_dim_out_2 + skip_2 + no_dim_2
unary_ufuncs = no_dim_1 + angle_1 + same_out + inv_angle_1 + special_dict + deg_rad + same_dim_in_1_nodim_out
implemented = (same_dim_out_2 + same_dim_in_2_nodim_out + same_dim_in_1_nodim_out + skip_2 + special_dict +
no_dim_1 + no_dim_2 + angle_1 + same_out + inv_angle_1 + deg_rad)


def quantify(x):
    if isinstance(x, Quantity):
        return x#.__copy__()
    else:
        return Quantity(x, DIMENSIONLESS)

def dimensionify(x):
    if isinstance(x, Dimension):
        return x
    elif isinstance(x, Quantity):
        return x.dimension
    elif np.isscalar(x) and not type(x) == str:
        return DIMENSIONLESS
    elif isinstance(x, np.ndarray):
        return DIMENSIONLESS
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
        return Quantity(x, DIMENSIONLESS, symbol=symbol, favunit=favunit)




class QuantityIterator(object):
    """General Quantity iterator (as opposed to flat iterator)"""
    def __init__(self, q):
        if not isinstance(q, Quantity):
            raise TypeError("QuantityIterator: must be Quantity object.")
        self.value = q.value
        self.dimension = q.dimension
        self.favunit = q.favunit
        if isinstance(q.value, np.ndarray):
            self.length = q.value.shape[0]
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
                             favunit=self.favunit)
            else:
                q_out = Quantity(self.value,
                             self.dimension,
                             favunit=self.favunit)
        self.count += 1

        return q_out

    
class FlatQuantityIterator(object):
    def __init__(self, q):
        self.value = q.value
        self.dimension = q.dimension
        self._flatiter = q.value.flat
        
    def __iter__(self):
        return self
    
    def __next__(self):
        value = next(self._flatiter)
        return Quantity(value, self.dimension)
    
    def __getitem__(self, indx):
        value = self._flatiter.__getitem__(indx)
        return Quantity(value, self.dimension)
        
    

def main():
    pass


if __name__ == "__main__":
    main()
