# !/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
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

About wrapping numpy functions :
https://odlgroup.github.io/odl/guide/numpy_guide.html
https://gist.github.com/shoyer/36b84ab064f027df318c0b823558de24
https://docs.scipy.org/doc/numpy/release.html#array-ufunc-added
http://www.numpy.org/neps/nep-0013-ufunc-overrides.html
http://www.astro.utoronto.ca/%7Emhvk/numpy-doc/neps/ufunc-overrides.html#proposed-interface
https://pint.readthedocs.io/en/0.9/numpy.html#comments
 
From physics : 
 - some packages relies on hand-parsing string or regex to parse fractions ? (could allow not relying on sympy)
 - use lambda and map to add/sub dicts of dimension power
 - Dimension object (PhysicalUnit) have a scale_factor and a offset attribute, that are multiplied as Dimensions are multiplie (and power are added), with conversion_factor/tuple method
 - add rpow method (must be dimensionless)
 - try to import uncertainties
 
 - allow to fix the default precision output of an instance
 - allow to fix an output format (as string) of an instance
 
From astropy comparison : 
- display about :
 - add a pretty display with latex
 - change display of powered units : m**2 to m2 ?
- other
 - allow conversion in different unit system ?
 - astropy converts value to float ?
 - should .to() be inplace ?  
 - declare a favunit for SI_units_derived ? 
 - keep symbols when * or / quantities ?
 - [X] : add imperial units from astropy. See units.
 - add cgs units from astropy ? See units.
 - add astrophys units from astropy ? See units.
 - find other common unit with same dimension ?
 - [X] : add SI units in units
 - add a string parser constructor ?
 - add a decorator for checking dimension
 - add a parser to favunit for functions output
 - Add method to simply return value in other systems
 - Deal with powers with fractions if necessary (utils)

From pint :
 - Allow adding units from a text file
 - Remplacer isinstance(quantity) par isinstance(self.__class__)
 - To() et ito()
 - Faire des property ?
 - Essayer de s’affranchir de numpy ?
 - Ajouter une propriété dimensionnalty qui renvoi les length
 - Ajouter un dict de dimneisonnaltiy classique comme acceleration, qui peut être consulté pour afficher la diemnsionnalty de façon plus sympa
 - Différencier in et to, l’un qui change juste la favunit, l’autre qui vérifier en plus que la favunit est de même dimension que l’objet
 - Prévoir un nom ? (m = Quantity(1, Dimension(« L »), name= ‘meter’))
 - Pouvoir spécifier un specifier par défaut ; Quantity.default_spec = « :s »
 - Utiliser __array_wrap__ pour overloader les fonctions numpy (see pint)
 - Check and refuse any « ^ » notation in string parsing ? only python power ** ? what about sympy ?
 - Method check(“dimensionality”) pour verifier à la main la dimension
 - Decorateur pour check dimension (accept Dimension object, Quantity, dimensionnality)
 - Ajouter methode plus_minus pour ajouter une incertitude
 - Able to define the units and base system in a file
 
From unum:
 - Quantify is a staticmethod, and defined as a decorator outside the class
 - Formatter is defined as a class, set_formet and reset_format as classmethod
 - Add a copy method with flags to include repr formats.
 - Add Fraction support for values
 - Store all quantities through their symbol, and check for Name conflict ?
 - Clean imports with del
 - Store quantities in a dict-like object ?
 - Use of Unicode superscripts ? display only, copy-paste problems ?
 
From magnitude : 
 - create a dictionnary tied to Quantity class, containing all the quantities ? units ? 
 - should Dimension be stored within the Quantity class (Quantity(1, kg=1, m=1)) ?
 - allow creating quantities with value and str of symbol of other quantity ?
 - init create oprec and oformat to None. They can be set later, and str and repr rely on the module value of oprec and oformat
 
From quantities :
 - Should Dimension allow addition (returning the same) ?
 - Copy ? hash ?
 - Dimension dict can drop unit if dimension is 0, or add a new unit (when multiplicating for ex)
 - Allow comparing dimension (??)
 - All numpy function and their corresponding checks are stored in a dict
 - Uses eval

Set constants value hard value
Add a handler to check if scipy is available, and update constants values verbosily
Add matplotlib compatibility
créer des entry points pour permettre la conversion de quantités en ligne de coommande
If a result dimension is equal to one of the 22 SI derived units, make it its favunit.
 


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

from .dimension import Dimension, DimensionError, SI_UNIT_SYMBOL

# # Constantes
UNIT_PREFIX= " "
DISPLAY_DIGITS = 2
EXP_THRESHOLD = 3
UNIT_SUFFIX = ""
#SCIENTIFIC = '%.' + str(DISPLAY_DIGITS) + 'E' # (syntaxe : "%.2f" % mon_nombre
#CLASSIC =  '%.' + str(DISPLAY_DIGITS) + 'f'

if DISPLAY_DIGITS < EXP_THRESHOLD:
    print("Warning: decimal display is less than exponent threshold")

# Numpy display options
np.set_printoptions(precision=DISPLAY_DIGITS,
                    threshold=EXP_THRESHOLD,
                    edgeitems=20)


#def turn_scalar_to_str(number,
#                       display_digit=DISPLAY_DIGITS,
#                       exp_thresh=EXP_THRESHOLD):
#    if isinstance(number,np.ndarray):
#        list_val_str = []
#        for val in number:
#            val_str = turn_scalar_to_str(val)
#            list_val_str = list_val_str + [val_str]
#        list_str = str(list_val_str)
#        list_str = list_str.replace(",","")
#        list_str = list_str.replace("'","")
#        return list_str
#    elif (isinstance(number,float) or isinstance(number,int) or
#          type(number) == np.int64 or type(number) == np.int32):
#        if np.all(np.isreal(number)):       # isrealobj()
#            if ((np.all(np.abs(number) >= 10**exp_thresh) or 
#                np.all(np.abs(number) < 10**(-exp_thresh))) and 
#                not number == 0):  # numpy.any(maCondition)
#                scientific = '%.' + str(display_digit) + 'E'
#                return scientific % number
#            else: 
#                classic =  '%.' + str(display_digit) + 'f'
#                return classic % number
#        elif np.any(np.iscomplexobj(number)):           
#            return ("(%s + %sj)" % (turn_scalar_to_str(number.real),
#                                    turn_scalar_to_str(number.imag)))
#        else:
#            raise TypeError("Number not real nor complex.")
#    else:
#        raise TypeError("Number must be array or number.")

# Decorator for trigo methods of Quantity object 
# This should be done via __array__ ?
def trigo_func(func):
    def func_dec(x):
        if not x.is_dimensionless_ext():
            raise DimensionError(x.dimension,
                                Dimension(None),
                                binary=False)
        return func(x)
    return func_dec

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
                super().__setattr__(name,value)
            else:
                raise TypeError(("Favorite unit of Quantity must be a Quantity "
                                 "or None, not {}").format(type(value)))
        else:
            super().__setattr__(name, value)
    
    # this breaks vectorize
    #def __getattr__(self,name):
    #    return self.value.__getattribute__(name)
        
    def __add__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension:                                                                                               
            raise DimensionError(self.dimension, y.dimension)                          
        return Quantity(self.value + y.value,
                        self.dimension)#, symbole = self.symbole)    

    def __radd__(self, x):
        return self + x

    def __sub__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension: 
            raise DimensionError(self.dimension, y.dimension)
        return Quantity(self.value - y.value,
                self.dimension)

    def __rsub__(self, x):
        x = quantify(x)
        return x - self

    def __mul__(self,y):
        y = quantify(y)
        return Quantity(self.value * y.value, 
                        self.dimension * y.dimension, 
                        symbol = self.symbol * y.symbol).remove_dimension_if_dimensionless() 
    
    __rmul__ = __mul__

    def __div__(self, y):
        y = quantify(y)
        return Quantity(self.value / y.value,
                        self.dimension / y.dimension,
                        symbol = self.symbol / y.symbol).remove_dimension_if_dimensionless()
    __truediv__ = __div__

    def __rdiv__(self, x):
        x = quantify(x)
        return x / self
    __rtruediv__ = __rdiv__

    def __floordiv__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return Quantity(self.value // y.value,
                       self.dimension).remove_dimension_if_dimensionless()
    
    def __mod__(self,y):
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return Quantity(self.value % y.value,
                       self.dimension).remove_dimension_if_dimensionless()

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
                        symbol = self.symbol ** power).remove_dimension_if_dimensionless()

    def __neg__(self):
        return self * (-1)

    def __invert__(self):
        return 1. / self
    __inv__ = __invert__

    def __len__(self):
        return len(self.value)

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

    def __ge__(self,y):
        return (self > y) | (self == y) # or bitwise

    def __le__(self,y):
        return (self < y) | (self == y) # or bitwise

    #def __and__()
    
    def __abs__(self):
        return Quantity(abs(self.value),
                        self.dimension,
                        symbol = self.symbol,
                        favunit = self.favunit)

    def __complex__(self):
        if not self.is_dimensionless():
            raise TypeError(("Quantity must be dimensionless "
                            "to be converted to complex."))
        return complex(self.value)
            
    def __int__(self):
        if not self.is_dimensionless():
            raise TypeError(("Quantity must be dimensionless "
                            "to be converted to int."))
        return int(self.value)

    def __float__(self):
        # This must be commented for solvers !
        #if not self.is_dimensionless():
        #    raise TypeError(("Quantity must be dimensionless "
        #                    "to be converted to float."))
        return float(self.value)
    
    def __round__(self, i):
        return Quantity(round(self.value, i), 
                       self.dimension,
                       favunit = self.favunit)

    def __repr__(self):
        return '<Quantity : ' + str(self.value) + " " + str(self.dimension.str_SI_unit()) + ">"        

    def __str__(self):
        complement_value_for_repr = self._compute_complement_value() 
        if not complement_value_for_repr == "":
            return str(self._compute_value()) + UNIT_PREFIX + complement_value_for_repr + UNIT_SUFFIX
        else: 
            return str(self._compute_value()) + UNIT_SUFFIX

    def __format_raw__(self, format_spec):
        return format(self.value, format_spec) + " " + str(self.dimension.str_SI_unit())

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

    # Pas trouvé d'utilité mais en a surement une
    def _SI_unitary_quantity(self):
        """Return a one-value quantity with same dimension.
        
        Such that self = self.value * self._SI_unitary_quantity
        """
        return Quantity(1, self.dimension)
    
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
    
    def _sum(self):
        """Extends numpy.sum to Quantity."""
        return Quantity(np.sum(self.value),
                       self.dimension,
                       favunit=self.favunit)

    def mean(self, **kwargs):
        """Extends numpy.mean to Quantity."""
        return Quantity(np.mean(self.value, **kwargs),
                       self.dimension,
                       favunit=self.favunit)
    
    def integrate(self, **kwargs):
        if not isinstance(self.value,np.ndarray):
            raise TypeError("Quantity value must be array-like to integrate.")
        return Quantity(np.trapz(self.value, **kwargs),
                       self.dimension,
                       favunit = self.favunit)
    
    def is_dimensionless(self):
        return self.dimension == Dimension(None)

    def remove_dimension_if_dimensionless(self):
        if self.is_dimensionless():
            return self.value          
        else:                           
            return self
        
    def has_integer_dimension_power(self):
        return all(is_integer(value) for value in self.dim.values())
    
    def to(self, y):
        """return quantity with another favunit."""
        if not isinstance(y, Quantity):
            raise TypeError("Cannot express Quantity in not Quantity")
        q = self
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
    def is_length(self):
        return self.dimension == Dimension("L")

    def is_surface(self):
        return self.dimension == Dimension("L")**2

    def is_volume(self):
        return self.dimension == Dimension("L")**3

    def is_time(self):
        return self.dimension == Dimension("T")

    def is_mass(self):
        return self.dimension == Dimension("M")

    def is_angle(self):
        return self.dimension == Dimension("RAD")

    def is_solid_angle(self):
        return self.dimension == Dimension("SR")

    def is_temperature(self):
        return self.dimension == Dimension("Θ")

    def is_dimensionless_ext(self):
        return self.is_dimensionless() or self.is_angle()

    @trigo_func
    def cos(self):
        return np.cos(self.value)
    
    @trigo_func
    def sin(self):
        return np.sin(self.value)
    @trigo_func
    def tan(self):
        return np.tan(self.value)
    
    @trigo_func
    def arccos(self):
        return np.arccos(self.value)
    
    @trigo_func
    def arcsin(self):
        return np.arcsin(self.value)
    
    @trigo_func
    def arctan(self):
        return np.arctan(self.value)
    
    

def quantify(x):
    if isinstance(x, Quantity):
        return x
    else:
        return Quantity(x,Dimension(None))
    

def make_quantity(x, symbol="UndefinedSymbol", favunit=None):
    if isinstance(x, Quantity):
        q = x
        q.symbol = symbol
        if favunit is None:
            q.favunit = x.favunit
        else:
            q.favunit = favunit
        return q
    else:
        return Quantity(x, Dimension(None), symbol=symbol, favunit=favunit)


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


class QuantityIterator(object):

    def __init__(self, q):
        if not isinstance(q, Quantity):
            raise TypeError("QuantityIterator: must be Quantity object.")
        self.value = q.value
        self.dimension = q.dimension
        self.favunit = q.favunit
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
                             favunit=self.favunit)
            else:
                q_out = Quantity(self.value,
                             self.dimension,
                             favunit=self.favunit)
        self.count += 1

        return q_out

    
    



def main():
    pass


if __name__ == "__main__":
    main()
