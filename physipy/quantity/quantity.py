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
from __future__ import annotations

import math
import numbers
import warnings
from typing import Callable, Union, Any

import numpy as np
import sympy as sp
import sympy.parsing as sp_parsing
import sympy.printing as sp_printing
from sympy import Expr, Symbol

from .dimension import DIMENSIONLESS, SI_UNIT_SYMBOL, Dimension, DimensionError

# # Constantes
UNIT_PREFIX = " "
DISPLAY_DIGITS = 2
EXP_THRESHOLD = 2
UNIT_SUFFIX = ""
LATEX_VALUE_UNIT_SEPARATOR = r"\,"  # " \cdot "
# DEFAULT_SYMBOL = sp.Symbol("UndefinedSymbol")
DEFAULT_SYMBOL = "UndefinedSymbol"
# SCIENTIFIC = '%.' + str(DISPLAY_DIGITS) + 'E' # (syntaxe : "%.2f" % mon_nombre
# CLASSIC =  '%.' + str(DISPLAY_DIGITS) + 'f'


VALUE_PROPERTY_BACKENDS: dict[type, dict] = {}


def register_property_backend(klass, interface_dict=None):
    """
    Idea to better handle value backends, like uncertainties.
    The idea is to provide to physipy a class and its interface dict,
    such that since Quantity looks at its values methods if item not found
    in the quantity itself

    Examples
    --------
    # BEFORE
    uv = Normal(1, 0.1, size=1000) # this is from class, with attribute .median()
    q = uv*m                       # this a quantity, with value of class
    q.median() # will return uv.median(), loosing the unit

    # AFTER
    import uncertainties as uc
    from physipy.quantity.quantity import register_property_backend

    uncertainties_property_backend_interface = {
        # res is the backend result of the attribute lookup, and q the wrapping quantity
        "nominal_value":lambda q, res: q._SI_unitary_quantity*res,
        "std_dev":lambda q, res: q._SI_unitary_quantity*res,
        "n":lambda q, res: q._SI_unitary_quantity*res,
        "s":lambda q, res: q._SI_unitary_quantity*res,
    }

    print("Registering uncertainties")
    register_property_backend(uc.core.Variable,
                            uncertainties_property_backend_interface)
                                register_value_backend(Normal, {"median":lambda res:res*self.unit})
    q.median() # will return uv.median()*m

    """
    if interface_dict is None:
        VALUE_PROPERTY_BACKENDS.pop(interface_dict)
    else:
        VALUE_PROPERTY_BACKENDS[klass] = interface_dict


class Quantity(object):
    """Quantity class :"""

    DIGITS = DISPLAY_DIGITS
    EXP_THRESH = EXP_THRESHOLD
    LATEX_SEP = LATEX_VALUE_UNIT_SEPARATOR

    # adding a __slots__ removes the presence of __dict__, which also
    # makes vars(q) to fail.
    # when this is commented, q.__dict__ returns something like :
    #    {'_value': 2.345,
    #     'dimension': <Dimension : {'L': 0, 'M': 0, 'T': -1, 'I': 0, 'theta': 1, 'N': 0, 'J': 0, 'RAD': 0, 'SR': 0}>,
    #     'symbol': 'toto',
    #     '_favunit': <Quantity : 1 s**2, symbol=s**2>}
    # using __dict__ instead of '_value', 'dimension', 'symbol', '_favunit' allows for both slots and
    # pickle to work (with __reduce__).
    __slots__ = "__dict__"

    def __init__(
        self,
        value,
        dimension: Dimension,
        symbol: Union[str, Symbol, Expr] = DEFAULT_SYMBOL,
        favunit: Quantity | None = None,
    ) -> None:
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

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        if isinstance(value, sp.Expr):
            self._symbol = value
        elif isinstance(value, str):
            self._symbol = sp.Symbol(value)
        else:
            raise TypeError(
                (
                    "Symbol of Quantity must be a string "
                    "or a sympy-symbol, "
                    "not {}"
                ).format(type(value))
            )

    @property
    def favunit(self):
        return self._favunit

    @favunit.setter
    def favunit(self, value):
        if isinstance(value, Quantity) or value is None:
            self._favunit = value
        # the following case is no tested, so removing it for now
        # elif np.isscalar(value):
        #     self._favunit = None
        else:
            raise TypeError(
                (
                    "Favorite unit of Quantity must be a Quantity "
                    "or None, not {}"
                ).format(type(value))
            )

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
        # return Quantity(self.value + y.value,
        #                self.dimension)
        return type(self)(self.value + y.value, self.dimension)

    def __radd__(self, x):
        return self + x

    def __sub__(self, y):
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return type(self)(self.value - y.value, self.dimension)

    def __rsub__(self, x):
        return quantify(x) - self

    def __mul__(self, y):
        # TODO make a decorator "try_raw_then_quantify_if_fail"
        try:
            return type(self)(
                self.value * y.value,
                self.dimension * y.dimension,
                symbol=self.symbol * y.symbol,
            ).rm_dim_if_dimless()
        except BaseException:
            y = quantify(y)
            return type(self)(
                self.value * y.value,
                self.dimension * y.dimension,
                symbol=self.symbol * y.symbol,
            ).rm_dim_if_dimless()

    __rmul__ = __mul__

    def __matmul__(self, y):
        y = quantify(y)
        return type(self)(
            self.value @ y.value,
            self.dimension * y.dimension,
            # symbol = self.symbol * y.symbol
        ).rm_dim_if_dimless()

    def __truediv__(self, y):
        y = quantify(y)
        return type(self)(
            self.value / y.value,
            self.dimension / y.dimension,
            symbol=self.symbol / y.symbol,
        ).rm_dim_if_dimless()

    def __rtruediv__(self, x):
        return quantify(x) / self

    def __floordiv__(self, y):
        """
        Any returned quantity should be dimensionless, but leaving the
        Quantity().remove() because more intuitive
        """
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return type(self)(
            self.value // y.value, self.dimension
        ).rm_dim_if_dimless()

    def __rfloordiv__(self, x):
        x = quantify(x)
        if not self.dimension == x.dimension:
            raise DimensionError(self.dimension, x.dimension)
        return type(self)(
            x.value // self.value, self.dimension
        ).rm_dim_if_dimless()

    def __mod__(self, y):
        """
        There is no rm_dim_if_dimless() because a
        modulo operation would not change the dimension.

        """
        y = quantify(y)
        if not self.dimension == y.dimension:
            raise DimensionError(self.dimension, y.dimension)
        return type(self)(
            self.value % y.value, self.dimension
        )  # .rm_dim_if_dimless()

    def __rpow__(power_self, base_other):
        base_other = quantify(base_other)
        if power_self.is_dimensionless:
            return Quantity(
                base_other.value**power_self.value,
                base_other.dimension ** float(power_self.value),
            ).rm_dim_if_dimless()
        raise TypeError

    def __pow__(self, power):
        """
        A power must always be a dimensionless scalar.
        If a = 1*m, we can't do a ** [1,2], because the result would be
        an array of quantity, and can't be a quantity with array-value,
        since the quantities won't be the same dimension.

        """
        # if not np.isscalar(power):#(isinstance(power,int) or isinstance(power,float)):
        #    raise TypeError(("Power must be a number, "
        #                    "not {}").format(type(power)))
        power = quantify(power).rm_dim_if_dimless()  # TODO : this feels ugly
        return type(self)(
            self.value**power,
            self.dimension**power,
            symbol=self.symbol**power,
        ).rm_dim_if_dimless()

    def __neg__(self):
        return Quantity(-self.value, self.dimension, favunit=self.favunit)

    def __pos__(self):
        return self

    def __len__(self):
        return len(self.value)

    def __bool__(self):
        return bool(self.value)

    # min and max uses the iterator
    def __min__(self):
        return Quantity(min(self.value), self.dimension, favunit=self.favunit)

    def __max__(self):
        return Quantity(max(self.value), self.dimension, favunit=self.favunit)

    def __eq__(self, y):
        # TODO : handle array comparison to return arrays
        try:
            y = quantify(y)
            return np.logical_and(
                (self.value == y.value), (self.dimension == y.dimension)
            )
        except Exception as e:
            return False

    def __ne__(self, y):
        # np.invert for element-wise not, for array compatibility
        return np.invert(self == y)

    def __gt__(self, y):
        y = quantify(y)
        if self.dimension == y.dimension:
            return self.value > y.value
        else:
            raise DimensionError(self.dimension, y.dimension)

    def __lt__(self, y):
        y = quantify(y)
        if self.dimension == y.dimension:
            return self.value < y.value
        else:
            raise DimensionError(self.dimension, y.dimension)

    def __ge__(self, y):
        return (self > y) | (self == y)  # or bitwise

    def __le__(self, y):
        return (self < y) | (self == y)  # or bitwise

    def __abs__(self):
        return type(self)(
            abs(self.value), self.dimension, favunit=self.favunit
        )

    def __complex__(self) -> complex:
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, DIMENSIONLESS, binary=False)
        return complex(self.value)

    def __int__(self) -> int:
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, DIMENSIONLESS, binary=False)
        return int(self.value)

    def __float__(self) -> float:
        if not self.is_dimensionless_ext():
            raise DimensionError(self.dimension, DIMENSIONLESS, binary=False)
        return float(self.value)

    def __round__(self, i=None):
        return type(self)(
            round(self.value, i), self.dimension, favunit=self.favunit
        )

    def __copy__(self):
        return type(self)(
            self.value,
            self.dimension,
            favunit=self.favunit,
            symbol=self.symbol,
        )

    def copy(self):
        return self.__copy__()

    def __repr__(self) -> str:
        if str(self.symbol) != "UndefinedSymbol":
            sym = ", symbol=" + str(self.symbol)
        else:
            sym = ""
        return (
            f"<{self.__class__.__name__} : "
            + str(self.value)
            + " "
            + str(self.dimension.str_SI_unit())
            + sym
            + ">"
        )

    def __str__(self) -> str:
        complement_value_for_repr = self._compute_complement_value()
        if not complement_value_for_repr == "":
            return (
                str(self._compute_value())
                + UNIT_PREFIX
                + complement_value_for_repr
                + UNIT_SUFFIX
            )
        else:
            return str(self._compute_value()) + UNIT_SUFFIX

    def __hash__(self):
        return hash(str(self.value) + str(self.dimension))

    def __reduce__(self):
        """
        Overload pickle behavior :
         - https://docs.python.org/3/library/pickle.html#object.__reduce__
         - https://stackoverflow.com/questions/19855156/whats-the-exact-usage-of-reduce-in-pickler
        TODO : form the doc :
            "Although powerful, implementing __reduce__() directly in your classes is
            error prone. For this reason, class designers should use the high-level
            interface (i.e., __getnewargs_ex__(), __getstate__() and __setstate__()) whenever possible.
        """
        return (
            self.__class__,
            (self.value, self.dimension, self.symbol, self.favunit),
        )

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

    # @property
    # def latex(self):
    #    return self._repr_latex_()

    # @property
    # def html(self):
    #    return self._repr_html_()

    # def _repr_pretty_(self, p, cycle):
    #    """Markdown hook for ipython repr.
    #    See https://ipython.readthedocs.io/en/stable/config/integrating.html"""
    #    print("repr_pretty")
    #    return p.text(self._repr_latex_())

    # TODO : assess the usefullness of this...
    def plot(self, kind: str = "y", other=None, ax=None) -> None:
        from physipy import plotting_context

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        with plotting_context():
            if kind == "y" and other is None:
                ax.plot(self)
            elif kind == "x" and other is not None:
                ax.plot(self, other)
            else:
                raise ValueError("kind must be y of x with other")

    def _repr_latex_(self) -> str:
        """Markdown hook for ipython repr in latex.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html"""
        try:
            # create a copy
            q = self.__copy__()
            # to set a favunit for display purpose
            # only change the favunit if not already defined
            if q.favunit is None:
                q.favunit = self._pick_smart_favunit()
            formatted_value = q._format_value()
            complemented = q._compute_complement_value()
            if complemented != "":
                # this line simplifies 'K*s/K' when a = 1*s and c = a.to(K)
                complement_value_str = sp_printing.latex(
                    sp_parsing.sympy_parser.parse_expr(complemented)
                )
            else:
                complement_value_str = ""
            # if self.value is an array, only wrap the complement in latex
            if isinstance(self.value, np.ndarray):
                return (
                    formatted_value
                    + "$"
                    + self.LATEX_SEP
                    + complement_value_str
                    + "$"
                )
            # if self.value is a scalar, use sympy to parse expression
            value_str = sp_printing.latex(
                sp_parsing.sympy_parser.parse_expr(formatted_value)
            )
            return (
                "$" + value_str + self.LATEX_SEP + complement_value_str + "$"
            )
        except Exception as e:
            # with some custom backend value, sympy has trouble parsing the
            # the expression: I'd rather have a regular string displayed rather
            # than an exception raised (since it is only used in notebook
            # context)
            return str(self)

    def _pick_smart_favunit(self, array_to_scal=np.mean) -> Quantity | None:
        """Method to pick the best favunit among the units dict.
        A smart favunit always have the same dimension as self.
        The 'best' favunit is the one minimizing the difference with self.
        In case self.value is an array, array_to_scal is
        used to convert the array to a single value.
        """
        from ._units import units
        from .utils import asqarray

        same_dim_unit_list = [
            value
            for value in units.values()
            if self.dimension == value.dimension
        ]
        # if no unit with same dim already exists
        if len(same_dim_unit_list) == 0:
            return None
        same_dim_unit_arr = asqarray(same_dim_unit_list)
        self_val = (
            self
            if not isinstance(self.value, np.ndarray)
            else array_to_scal(self)
        )
        best_ixd = np.abs(same_dim_unit_arr - np.abs(self_val)).argmin()
        best_favunit = same_dim_unit_list[best_ixd]
        return best_favunit

    def _format_value(self) -> str:
        """Used to format the value on repr as a str.
        If the value is > to 10**self.EXP_THRESH, it is displayed with scientific notation.
        Else floating point notation is used.
        """
        value: Any = self._compute_value()
        if not isinstance(value, numbers.Number):  # np.isscalar(value):
            return str(value)
        else:
            if abs(value) >= 10 ** self.EXP_THRESH or abs(value) < 10 ** (
                -self.EXP_THRESH
            ):
                return ("{:." + str(self.DIGITS) + "e}").format(value)
            else:
                return ("{:." + str(self.DIGITS) + "f}").format(value)

    # def _repr_markdown_(self):
    #    """Markdown hook for ipython repr in markdown.
    #    this seems to take precedence over _repr_latex_"""
    #    return self.__repr__()

    # def _repr_html(self):
    #    return self._repr_latex_()

    # def __format_raw__(self, format_spec):
    # return format(self.value, format_spec) + " " +
    # str(self.dimension.str_SI_unit())

    def __format__(self, format_spec: str) -> str:
        """This method is used when using format or f-string.
        The format is applied to the numerical value part only."""
        complement_value_for_repr = self._compute_complement_value()
        prefix = UNIT_PREFIX
        if "~" in format_spec:
            format_spec = format_spec.replace("~", "")
            prefix = ""
        if not complement_value_for_repr == "":
            return (
                format(self._compute_value(), format_spec)
                + prefix
                + complement_value_for_repr
                + UNIT_SUFFIX
            )
        else:
            return format(self._compute_value(), format_spec) + prefix

    def _compute_value(self):
        """Return the numerical value corresponding to favunit."""
        if isinstance(self.favunit, Quantity):
            ratio_favunit = make_quantity(self / self.favunit)
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
            ratio_favunit = make_quantity(self / favunit)
            dim_SI = ratio_favunit.dimension
            if dim_SI == DIMENSIONLESS:
                return str(favunit.symbol)
            else:
                return str(favunit.symbol) + "*" + dim_SI.str_SI_unit()
        else:
            return self.dimension.str_SI_unit()

    # used for plotting
    @property
    def _SI_unitary_quantity(self):
        """Return a one-value quantity with same dimension.

        Such that self = self.value * self._SI_unitary_quantity
        """
        return type(self)(
            1, self.dimension, symbol=self.dimension.str_SI_unit()
        )

    def __getitem__(self, idx):
        """
        Having this defined here makes iter(m) not raising an exception
        so np.iterable considers m as iterable, which is not.
        Maybe use monkey patching ?
        Solution was to define __iter__ since iter first checks that
        x.__iter__ doesn't raise a TypeError
        """
        return type(self)(
            self.value[idx], self.dimension, favunit=self.favunit
        )

    def __setitem__(self, idx, q) -> None:
        q = quantify(q)
        if not q.dimension == self.dimension:
            raise DimensionError(q.dimension, self.dimension)
        if isinstance(idx, np.bool_) and idx:
            self.valeur = q.value
        elif isinstance(idx, np.bool_) and idx is False:
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
        if isinstance(self.value, np.ndarray):
            return QuantityIterator(self)
        else:
            return iter(self.value)

    @property
    def flat(self) -> FlatQuantityIterator:
        # pint implementation
        # for v in self.value.flat:
        #    yield Quantity(v, self.dimension)

        # astropy
        return FlatQuantityIterator(self)

    def flatten(self):
        return type(self)(
            self.value.flatten(), self.dimension, favunit=self.favunit
        )

    def tolist(self) -> list:
        return [type(self)(i, self.dimension) for i in self.value]

    @property
    def real(self):
        return type(self)(self.value.real, self.dimension, favunit=self.favunit)

    @property
    def imag(self):
        return type(self)(self.value.imag, self.dimension, favunit=self.favunit)

    def conjugate(self):
        return type(self)(self.value.conjugate(), self.dimension, favunit=self.favunit)

    @property
    def T(self):
        return type(self)(self.value.T, self.dimension, favunit=self.favunit)

    def inverse(self):
        """is this method usefull ?"""
        return type(self)(1 / self.value, 1 / self.dimension)

    # see list of array methods:
    # https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-methods
    def min(self, **kwargs):
        return np.min(self, **kwargs).set_favunit(self.favunit)

    def max(self, **kwargs):
        return np.max(self, **kwargs).set_favunit(self.favunit)

    def sum(self, **kwargs):
        return np.sum(self, **kwargs).set_favunit(self.favunit)

    def mean(self, **kwargs):
        return np.mean(self, **kwargs).set_favunit(self.favunit)

    def std(self, *args, **kwargs):
        return np.std(self, *args, **kwargs).set_favunit(self.favunit)

    def conj(self):
        return np.conj(self).set_favunit(self.favunit)

    def round(self, *args, **kwargs):
        return np.round(self, *args, **kwargs).set_favunit(self.favunit)

    def var(self, **kwargs):
        return np.var(self, **kwargs)

    def abs(self):
        return np.abs(self)

    def integrate(self, *args, **kwargs):
        return np.trapz(self, *args, **kwargs)

    def is_dimensionless(self) -> bool:
        return self.dimension == DIMENSIONLESS

    def rm_dim_if_dimless(self):
        if self.is_dimensionless():
            return self.value
        else:
            return self

    def has_integer_dimension_power(self) -> bool:
        return all(
            value == int(value) for value in self.dimension.dim_dict.values()
        )

    def to(self, y: Quantity):
        """return quantity with another favunit."""
        if not isinstance(y, Quantity):
            raise TypeError("Cannot express Quantity in not Quantity")
        q = self.__copy__()
        q.favunit = y
        return q

    def set_favunit(self, fav: Quantity) -> Quantity:
        """
        To be used as one-line declaration : (np.linspace(3, 10)*mum).set_favunit(mum)
        """
        self.favunit = fav
        return self

    def set_symbol(self, symbol: str) -> Quantity:
        """
        To be used as one-line declaration for a favunit : my_period=(10*s).set_symbol("period")
        """
        self.symbol = symbol
        return self

    def into(self, y: Quantity) -> Quantity:
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
    def is_length(self) -> bool:
        return self.dimension == Dimension("L")

    def is_surface(self) -> bool:
        return self.dimension == Dimension("L") ** 2

    def is_volume(self) -> bool:
        return self.dimension == Dimension("L") ** 3

    def is_time(self) -> bool:
        return self.dimension == Dimension("T")

    def is_mass(self) -> bool:
        return self.dimension == Dimension("M")

    def is_angle(self) -> bool:
        return self.dimension == Dimension("RAD")

    def is_solid_angle(self) -> bool:
        return self.dimension == Dimension("SR")

    def is_temperature(self) -> bool:
        return self.dimension == Dimension("theta")

    def is_nan(self):
        "For use with pandas extension"
        return np.isnan(self.value)

    def is_dimensionless_ext(self) -> bool:
        return self.is_dimensionless() or self.is_angle()

    def check_dim(self, dim) -> bool:
        return self.dimension == dimensionify(dim)

    # for munits support
    def _plot_get_value_for_plot(self, q_unit):
        q_unit = quantify(q_unit)
        if not self.dimension == q_unit.dimension:
            raise DimensionError(self.dimension, q_unit.dimension)
        return self / q_unit

    # for munits support
    def _plot_extract_q_for_axe(self, units_list):
        if self.favunit is None:
            favs = [
                unit
                for unit in units_list
                if self._SI_unitary_quantity == unit
            ]
            if len(favs) >= 1:
                favunit = favs[0]
            else:
                favunit = self.favunit
        else:
            favunit = self.favunit
        if isinstance(favunit, Quantity):
            ratio_favunit = make_quantity(self / favunit)
            dim_SI = ratio_favunit.dimension
            if dim_SI == DIMENSIONLESS:
                return favunit
            else:
                return make_quantity(
                    favunit * ratio_favunit._SI_unitary_quantity,
                    symbol=str(favunit.symbol)
                    + "*"
                    + ratio_favunit._SI_unitary_quantity.dimension.str_SI_unit(),
                )
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
        # if item == '__iter__':
        #   if isinstance(self.value,np.ndarray):
        #        return QuantityIterator(self)
        #    else:
        #        return iter(self.value)
        if item.startswith("__array_"):
            warnings.warn(f"The unit of the quantity is stripped for {item}")
            if isinstance(self.value, np.ndarray):
                return getattr(self.value, item)
            else:
                # If an `__array_` attributes is requested but the magnitude is not an ndarray,
                # we convert the magnitude to a numpy ndarray.
                self.value = np.array(self.value)
                return getattr(self.value, item)
        try:
            # This block allows to customize specific value-backend attributes
            # like "nominale_value" when using uncertainties using :
            # import uncertainties as uc
            # from physipy.quantity.quantity import register_property_backend
            #
            # uncertainties_property_backend_interface = {
            #     # res is the backend result of the attribute lookup, and q the wrapping quantity
            #     "nominal_value":lambda q, res: q._SI_unitary_quantity*res,
            #     "std_dev":lambda q, res: q._SI_unitary_quantity*res,
            #     "n":lambda q, res: q._SI_unitary_quantity*res,
            #     "s":lambda q, res: q._SI_unitary_quantity*res,
            # }
            #
            # register_property_backend(
            #     uc.core.Variable,
            #     uncertainties_property_backend_interface
            # )

            if type(self.value) in VALUE_PROPERTY_BACKENDS:
                interface = VALUE_PROPERTY_BACKENDS[type(self.value)]
                if item in interface:
                    res = getattr(self.value, item)
                    return interface[item](self, res)
            res = getattr(self.value, item)
            return res
        except AttributeError as ex:
            raise AttributeError(
                "Neither Quantity object nor its value ({}) "
                "has attribute '{}'".format(self.value, item)
            )

    # def to_numpy(self):
    #    """
    #    Needed for plt.hist(np.arange(10)*m).
    #    """
    #    return np.asarray(self.value)

    def reshape(self, *args, **kwargs):
        return type(self)(self.value.reshape(*args, **kwargs), self.dimension)

    def __array_function__(self, func, types, args, kwargs):
        from ._numpy import HANDLED_FUNCTIONS

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        # if not all(issubclass(t, self.__class__) for t in types):
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
            raise NotImplementedError(
                f"array ufunc {ufunc} with method {method} not implemented"
            )

    def _ufunc_accumulate(self, ufunc, method, *args, **kwargs):
        ufunc_name = ufunc.__name__
        left = args[0]
        # hypot doesn't have a reduce
        if ufunc_name in ["add"]:
            res = ufunc.accumulate(left.value, **kwargs)
            return type(self)(res, left.dimension)
        else:
            raise NotImplementedError(
                f"array ufunc {ufunc} with method {method} not implemented"
            )

    def _ufunc_reduce(self, ufunc, method, *args, **kwargs):
        """
        The method == "reduce" part of __array_ufunc__ interface.
        """
        if not method == "reduce":
            raise NotImplementedError(
                f"array ufunc {ufunc} with method {method} not implemented"
            )
        ufunc_name = ufunc.__name__
        left = args[0]  # removed quantify...
        from ._numpy import (
            angle_1,
            inv_angle_1,
            no_dim_1,
            no_dim_2,
            same_dim_in_1_nodim_out,
            same_dim_in_2_nodim_out,
            same_dim_out_2,
            same_out,
            skip_2,
            special_dict,
            unary_ufuncs,
        )

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
                raise NotImplementedError(
                    f"array ufunc {ufunc} with method {method} not implemented"
                )
        # return booleans :
        elif ufunc_name in same_dim_in_2_nodim_out:
            res = ufunc.reduce(left.value, **kwargs)
            return res
        # ValueError: reduce only supported for binary functions
        elif ufunc_name in unary_ufuncs:
            raise ValueError("reduce only supported for binary functions.")
        else:
            raise NotImplementedError(
                f"array ufunc {ufunc} with method {method} not implemented"
            )

    def _ufunc_call(self, ufunc, method, *args, **kwargs):
        """
        The method == "__call__" part of __array_ufunc__ interface.
        """
        if not method == "__call__":
            raise NotImplementedError(
                f"array ufunc {ufunc} with method {method} not implemented"
            )
        ufunc_name = ufunc.__name__
        left = quantify(args[0])
        from ._numpy import (
            angle_1,
            inv_angle_1,
            no_dim_1,
            no_dim_2,
            same_dim_in_1_nodim_out,
            same_dim_in_2_nodim_out,
            same_dim_out_2,
            same_out,
            skip_2,
            special_dict,
        )

        if ufunc_name in same_dim_out_2:
            other = quantify(args[1])
            if not left.dimension == other.dimension:
                raise DimensionError(left.dimension, other.dimension)
            res = ufunc.__call__(left.value, other.value)
            return type(self)(res, left.dimension)
        elif ufunc_name in skip_2:
            other = quantify(args[1])
            res = ufunc.__call__(left.value, other.value)
            if ufunc_name == "multiply" or ufunc_name == "matmul":
                return type(self)(res, left.dimension * other.dimension)
            elif ufunc_name == "divide" or ufunc_name == "true_divide":
                return type(self)(
                    res, left.dimension / other.dimension
                ).rm_dim_if_dimless()
            elif ufunc_name == "copysign" or ufunc_name == "nextafter":
                return type(self)(res, left.dimension)
        elif ufunc_name in no_dim_1:
            if not left.dimension == DIMENSIONLESS:
                raise DimensionError(left.dimension, DIMENSIONLESS)
            res = ufunc.__call__(left.value)
            return type(self)(res, DIMENSIONLESS)
        elif ufunc_name in angle_1:
            if not left.is_dimensionless_ext():
                raise DimensionError(
                    left.dimension, DIMENSIONLESS, binary=True
                )
            res = ufunc.__call__(left.value)
            return type(self)(res, DIMENSIONLESS).rm_dim_if_dimless()
        elif ufunc_name in same_out:
            res = ufunc.__call__(left.value)
            return type(self)(res, left.dimension).rm_dim_if_dimless()
        elif ufunc_name in special_dict:
            if ufunc_name == "sqrt":
                res = ufunc.__call__(left.value)
                return type(self)(res, left.dimension ** (1 / 2))
            elif ufunc_name == "power":
                power_num = args[1]
                if not (
                    isinstance(power_num, int) or isinstance(power_num, float)
                ):
                    raise TypeError(
                        ("Power must be a number, " "not {}").format(
                            type(power_num)
                        )
                    )
                res = ufunc.__call__(left.value, power_num)
                return type(self)(
                    res,
                    left.dimension**power_num,
                    symbol=left.symbol**power_num,
                ).rm_dim_if_dimless()
            elif ufunc_name == "reciprocal":
                res = ufunc.__call__(left.value)
                return type(self)(res, 1 / left.dimension)
            elif ufunc_name == "square":
                res = ufunc.__call__(left.value)
                return type(self)(res, left.dimension**2)
            elif ufunc_name == "cbrt":
                res = ufunc.__call__(left.value)
                return type(self)(res, left.dimension ** (1 / 3))
            elif ufunc_name == "modf":
                res = ufunc.__call__(left.value)
                frac, integ = res
                return (
                    type(self)(frac, left.dimension),
                    type(self)(integ, left.dimension),
                )
            elif ufunc_name == "arctan2":
                # both x and y should have same dim such that the ratio is
                # dimless
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
        # elif ufunc_name in inv_angle_2:
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
            if not (
                left.dimension == DIMENSIONLESS
                and other.dimension == DIMENSIONLESS
            ):
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


def quantify(x):
    if isinstance(x, Quantity):
        return x  # .__copy__()
    else:
        return Quantity(x, DIMENSIONLESS)


def dimensionify(x) -> Dimension:
    if isinstance(x, Dimension):
        return x
    elif isinstance(x, Quantity):
        return x.dimension
    elif np.isscalar(x) and not isinstance(x, str):
        # mostly to handle x = 1
        return DIMENSIONLESS
    elif isinstance(x, np.ndarray):
        return DIMENSIONLESS
    else:
        return Dimension(x)


def make_quantity(
    x, symbol="UndefinedSymbol", favunit: Quantity | None = None
) -> Quantity:
    """
    Create a new Quantity from x, with optionnal symbol and favunit.
    If x is already a Quantity, also copy its favunit if favunit=None.
    If x is not a Quantity, create a dimensionless Quantity.
    """
    if isinstance(x, Quantity):
        q = x.__copy__()
        q.symbol = symbol
        if favunit is None:
            if x.favunit is not None:
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
                q_out = Quantity(
                    self.value[self.count],
                    self.dimension,
                    favunit=self.favunit,
                )
            else:
                q_out = Quantity(
                    self.value, self.dimension, favunit=self.favunit
                )
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
