# coding: utf-8
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
 - [ ] : create a Wrong dimension Error, for trigo functions for eg
 - [X] : deal with numpy slicing a[a>1]
 - [ ] : improve integration of eq, ne (ex : assertNotEqual when dealing with arrays)
 - [ ] : when uncertainties is implemented, add an automatic plotting
 
From astropy comparison : 
- display about :
 - [ ] : display digits '1.' instead of 1.00
 - add a pretty display with latex
 - display default number of digits ?
 - [ ] : add a format method --> need a refactor of repr..
 - change display of powered units : m**2 to m2 ?
- other
 - allow conversion in different unit system ?
 - astropy converts value to float ?
 - should .to() be inplace ?  
 - declare a favunit for SI_units_derived ? 
 - keep symbols when * or / quantities ?
 - [X] : add imperial units from astropy
 - add cgs units from astropy ?
 - add astrophys units from astropy ?
 - find other common unit with same dimension ?
 - [X] : add SI units in units
 - add a string parser constructor ?

 

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


PENDING : 
 - rmul, truediv, rtruediv
 - modification in iterator next
 - doit être dimensionless pour conversion en float
 - decorator for trigo methods

"""

import sys
sys.path.insert(0,'/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/dimension')

import scipy
import scipy.integrate
import scipy.optimize

import numpy as np
import sympy as sp
import numbers as nb

from dimension import Dimension, DimensionError, SI_UNIT_SYMBOL


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


def turn_scalar_to_str(number,
                       display_digit=DISPLAY_DIGITS,
                       exp_thresh=EXP_THRESHOLD):
    if isinstance(number,np.ndarray):
        list_val_str = []
        for val in number:
            val_str = turn_scalar_to_str(val)
            list_val_str = list_val_str + [val_str]
        list_str = str(list_val_str)
        list_str = list_str.replace(",","")
        list_str = list_str.replace("'","")
        return list_str
    elif (isinstance(number,float) or isinstance(number,int) or
          type(number) == np.int64 or type(number) == np.int32):
        if np.all(np.isreal(number)):       # isrealobj()
            if ((np.all(np.abs(number) >= 10**exp_thresh) or 
                np.all(np.abs(number) < 10**(-exp_thresh))) and 
                not number == 0):  # numpy.any(maCondition)
                scientific = '%.' + str(display_digit) + 'E'
                return scientific % number
            else: 
                classic =  '%.' + str(display_digit) + 'f'
                return classic % number
        elif np.any(np.iscomplexobj(number)):           
            return ("(%s + %sj)" % (turn_scalar_to_str(number.real),
                                    turn_scalar_to_str(number.imag)))
        else:
            raise TypeError("Number not real nor complex.")
    else:
        raise TypeError("Number must be array or number.")

# Decorator for trigo methods of Quantity object 
# This should be done via __array__ ?
def trigo_func(func):
    def func_dec(x):
        if not x.is_dimensionless_ext():
            raise TypeError(("Quantity must be dimensionless or "
                             "angle to compute trigonometric value"))
        return func(x)
    return func_dec

class Quantity(object):
    """Quantity class : """
    
    DIGITS = DISPLAY_DIGITS
    EXP_THRESH = EXP_THRESHOLD
    
    def __init__(self,value,dimension,symbol="UndefinedSymbol",favunit=None):
        self.__array_priority__ = 100
        self.value = value
        self.dimension = dimension
        self.symbol = symbol
        self.favunit = favunit

    def __setattr__(self,name,value):
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
        
    def __add__(self,y):
        y = quantify(y)
        if not self.dimension == y.dimension:                                                                                               
            raise DimensionError(self.dimension,y.dimension)                          
        return Quantity(self.value + y.value,
                        self.dimension)#, symbole = self.symbole)    

    def __radd__(self,x):
        return self + x

    def __sub__(self,y):
        y = quantify(y)
        if not self.dimension == y.dimension: 
            raise DimensionError(self.dimension,y.dimension)
        return Quantity(self.value - y.value,
                self.dimension)

    def __rsub__(self,x):
        x = quantify(x)
        return x - self

    def __mul__(self,y):
        y = quantify(y)
        return Quantity(self.value * y.value, 
                        self.dimension * y.dimension, 
                        symbol = self.symbol * y.symbol).remove_dimension_if_dimensionless() 
    __rmul__ = __mul__
   # def __rmul__(self,x):
   #     return self * x

    def __div__(self,y):
        y = quantify(y)
        return Quantity(self.value / y.value,
                        self.dimension / y.dimension,
                        symbol = self.symbol / y.symbol).remove_dimension_if_dimensionless()
    __truediv__ = __div__

    def __rdiv__(self,x):
        x = quantify(x)
        return x / self
    __rtruediv__ = __rdiv__

    
    def __floordiv__(self,y):
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
    
    # def __truediv__(self,y):
    #    return self.__div__(y)

    # def __rtruediv__(self,x):
    #    return self.__rdiv__(x)

    def __pow__(self,power):
        """
        A power must always be a dimensionless scalar. 
        If a = 1*m, we can't do a ** [1,2], because the result would be
        an array of quantity, and can't be a quantity with array-value, 
        since the quantities won't be the same dimension.
        
        """
        if not (isinstance(power,int) or isinstance(power,float)):
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
            #return self.value == y.value
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
        #return round(self.value, i)

    def __str__(self):
        return self.__repr__()
    
    #def __format__(self, format_spec):
    #    return 

    #####################################################
    # TODO : factorize this with repr
    #def value_with_favunit_if_possible(self):
    #    if isinstance(self.favunit, Quantity):
    #        return make_quantity(self/self.favunit).value
    #    else:
    #        return self.value
    # TODO : factorize this with repr
    #def unit_repr_with_favunit(self):
    #    if isinstance(self.favunit, Quantity):
    #        ratio_favunit = make_quantity(self/self.favunit)
    #        dim_SI = ratio_favunit.dimension
    #        str_favunit = str(self.favunit.symbol) + "*" + dim_SI.str_SI_unit()
    #        return str_favunit
    #    else:
    #        return self.dimension.str_SI_unit()
    
    # TODO : factorize this with repr
    def _compute_value(self):
        """Return the numerical value corresponding to favunit."""
        if isinstance(self.favunit, Quantity):
            ratio_favunit = make_quantity(self/self.favunit)
            return ratio_favunit.value
        else:
            return self.value
    
    # TODO : factorize this with repr
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
    
    def __repr__(self):
        value_for_repr = self._compute_value()
        complement_value_for_repr = self._compute_complement_value()
        if not complement_value_for_repr == "":
            return turn_scalar_to_str(value_for_repr) + UNIT_PREFIX + complement_value_for_repr + UNIT_SUFFIX
        else: 
            return turn_scalar_to_str(value_for_repr) + UNIT_SUFFIX
    
    # Gardé pour test
    #def _old_repr(self):
    #    if isinstance(self.favunit, Quantity):
    #        return self.__repr_with_favunit__()
    #    else:
    #        return (turn_scalar_to_str(self.value) + UNIT_PREFIX + 
    #                self.dimension.str_SI_unit() + UNIT_SUFFIX)
    
    # gardé pour test
    #def __repr_with_favunit__(self):
    #    ratio_favunit = self / self.favunit
    #    if isinstance(ratio_favunit, Quantity):
    #        value_SI = ratio_favunit.value
    #        dim_SI = ratio_favunit.dimension
    #        str_favunit = (str(self.favunit.symbol) + "*" 
    #                       + dim_SI.str_SI_unit())
    #    else:
    #        value_SI = ratio_favunit
    #        str_favunit = str(self.favunit.symbol)
    #    return (turn_scalar_to_str(value_SI) + " " + str_favunit + UNIT_SUFFIX)

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
        """in-place change of favunit."""
        if not isinstance(y, Quantity):
            raise TypeError("Cannot express Quantity in not Quantity")
        #q = self
        #q.favunit = y
        #return q
        self.favunit = y
        return self
    
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
        return (self.dimension == Dimension("RAD") or 
                self.dimension == Dimension("SR"))
    def is_temperature(self):
        return self.dimension == Dimension("Θ")
    def is_dimensionless_ext(self):
        return self.is_dimensionless() or self.is_angle()

#    def qplot(self):
#        if not isinstance(value,np.ndarray):
#            raise TypeError("Value must be array-like to be plotted.")
#        try:
#            import matplotlib.pyploy as plt
#            plt.plot(self.value)
#            plt.show()

    @trigo_func
    def cos(self):
        #if not self.is_dimensionless_ext():
        #    raise TypeError(("Quantity must be dimensionless or "
        #                     "angle to compute trigonometric value"))
        return np.cos(self.value)
    @trigo_func
    def sin(self):
        #if not self.is_dimensionless_ext():
        #    raise TypeError(("Quantity must be dimensionless or "
        #                     "angle to compute trigonometric value"))
        return np.sin(self.value)
    @trigo_func
    def tan(self):
        #if not self.is_dimensionless_ext():
        #    raise TypeError(("Quantity must be dimensionless or "
        #                     "angle to compute trigonometric value"))
        return np.tan(self.value)
    @trigo_func
    def arccos(self):
        #if not self.is_dimensionless_ext():
        #    raise TypeError(("Quantity must be dimensionless or "
        #                     "angle to compute trigonometric value"))
        return np.arccos(self.value)
    @trigo_func
    def arcsin(self):
        #if not self.is_dimensionless_ext():
        #    raise TypeError(("Quantity must be dimensionless or "
        #                     "angle to compute trigonometric value"))
        return np.arcsin(self.value)
    @trigo_func
    def arctan(self):
        #if not self.is_dimensionless_ext():
        #    raise TypeError(("Quantity must be dimensionless or "
        #                     "angle to compute trigonometric value"))
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

    
    


# TRIGO
def sqrt(x):
    x = quantify(x)
    return Quantity(np.sqrt(x.value), x.dimension**(1/2))

# Trigo moved to class-method
#def cos(x):
#    x = quantify(x)
#    if not x.is_dimensionless_ext():
#        raise DimensionError
#    return np.cos(x.value)
#
#def sin(x):
#    x = quantify(x)
#    if not x.is_dimensionless_ext():
#        raise DimensionError
#    return np.sin(x.value)
#
#def tan(x):
#    x = quantify(x)
#    if not x.is_dimensionless_ext():
#        raise DimensionError
#    return np.tan(x.value)
#
#def arccos(x):
#    x = quantify(x)
#    if not x.is_dimensionless_ext():
#        raise DimensionError
#    return np.arccos(x.value)
#
#def arcsin(x):
#    x = quantify(x)
#    if not x.is_dimensionless_ext():
#        raise DimensionError
#    return np.arcsin(x.value)
#
#def arctan(x):
#    x = quantify(x)
#    if not x.is_dimensionless_ext():
#        raise DimensionError
#    return np.arctan(x.value)

# Generiques
def linspace(Q_1, Q_2, nb_points=100):
    """Generate a lineary-spaced vector of Quantity.
    
    This function aims to extend numpy.linspace to Quantity objects.
    
    """
    Q_1 = quantify(Q_1)
    Q_2 = quantify(Q_2)
    if not Q_1.dimension == Q_2.dimension:
        raise DimensionError(Q_1.dimension, Q_2.dimension)
    val_out = np.linspace(Q_1.value, Q_2.value, nb_points)
    dim_out = Q_1.dimension
    favunit_out = Q_1.favunit
    return Quantity(val_out,
                    dim_out,
                    favunit=favunit_out)#.remove_dimension_if_dimensionless()


def interp(x, tab_x, tab_y):
    """Interpolate the value of x in tab_y based on tab_x.
    
    This function aims to extend numpy.interp to Quantity.
    
    """
    x = quantify(x)
    tab_x = quantify(tab_x)
    tab_y = quantify(tab_y)
    if not x.dimension == tab_x.dimension:
        raise DimensionError(x, tab_x)
    val_interp = np.interp(x.value, tab_x.value, tab_y.value)
    dim_interp = tab_y.dimension
    favunit_interp = tab_y.favunit
    return Quantity(val_interp,
                    dim_interp,
                    favunit=favunit_interp)#.remove_dimension_if_dimensionless()


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
                    dim_trapz).remove_dimension_if_dimensionless()


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
    return Quantity(integral, dim_out)#.remove_dimension_if_dimensionless()


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
                   res_dim * x0.dimension).remove_dimension_if_dimensionless(), prec


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
                   res_dim * x0.dimension * y0.dimension).remove_dimension_if_dimensionless(), prec


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
                   res_dim * x0.dimension * y0.dimension * z0.dimension).remove_dimension_if_dimensionless(), prec    


# Generique 
def qroot(func_cal, start):
    start_val = start.value
    start_dim = start.dimension
    def func_cal_float(x_float):
        return func_cal(Quantity(x_float,start_dim))
    return Quantity(scipy.optimize.root(func_cal_float, start_val).x[0], start_dim) #♦Quantity(fsolve(func_cal_float, start_val), start_dim)

def qbrentq(func_cal, start, stop):
    start_val = start.value
    stop_val = stop.value
    start_dim = start.dimension
    def func_cal_float(x_float):
        return func_cal(Quantity(x_float,start_dim))
    return Quantity(scipy.optimize.brentq(func_cal_float, start_val, stop_val), start_dim) #♦Quantity(fsolve(func_cal_float, start_val), start_dim)


def main():
    pass


if __name__ == "__main__":
    main()
