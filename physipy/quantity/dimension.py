# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""allows manipulating physical Dimension objects.

TODO:
 * [ ] : add a docstring.
 * [X] : better define what pow method accepts : leaving isreal for now...
 * [X] : add a method to invert the Dimension
 * [X] : rename the method without the dunders (not for special methods)
 * [X] : declare truediv as equal to div (and so on).
 * [X] : make exception strings display correctly
 * [X] : char for temperatures Θ is changed to "theta" (for sympy)
 * [ ] : try a cleaner convesrion from dict to str

PROPOSITIONS:
 * method to return a latex-formated str ?
 * change the str/repr style to a table-view of the dimension content ?
 * should sr be just a unit with dimension rad**2 ?
 * add a full-named repr ? (ex: "length/time")
 * change the dimension representation from dict to array (faster)
 * allow construction with strings (Dimension("m**2") or Dimension ("L**2")) ?

PLEASE NOTE :
rad and sr are not base SI-units, but were added for convenience.
They can be deleted if not needed, but update tests in consequence.

"""

import sympy as sp
import numpy as np

SI_UNIT_SYMBOL = {
    'L': 'm',
    'M': 'kg',
    'T': 's',
    'I': 'A',
    'theta': 'K',
    'N': 'mol',
    'J': 'cd',
    'RAD': 'rad',
    'SR': 'sr',
}

SI_SYMBOL_LIST = SI_UNIT_SYMBOL.keys()
NO_DIMENSION_STR = "no-dimension"


class DimensionError(Exception):
    """Exception class for dimension errors."""

    def __init__(self, dim_1, dim_2, binary=True):
        """Init method of DimensionError class."""
        if binary:
            self.message = ("Dimension error : dimensions of "
                            "operands are {} and {}, and are "
                            "differents.").format(str(dim_1), str(dim_2))
        else:
            self.message = ("Dimension error : dimension is {} "
                            "but should be {}").format(str(dim_1), str(dim_2))

    def __str__(self):
        """Str method of DimensionError class."""
        return self.message


class Dimension(object):
    """Allows to manipulate physical dimensions."""

    def __init__(self, definition):
        """Allow the creation of Dimension object with 3 possibile ways."""
        self.dim_dict = {}
        for dim_symbol in SI_SYMBOL_LIST:
            self.dim_dict[dim_symbol] = 0
        if definition is None:
            pass  # dimension_symbol_dict already initialized
        elif definition in list(self.dim_dict.keys()):
            self.dim_dict[definition] = 1
        elif (isinstance(definition, dict) and
              set(list(definition.keys())).issubset(SI_SYMBOL_LIST)):
            for dim_symbol, dim_power in definition.items():
                if dim_power == int(dim_power):
                    dim_power = int(dim_power)
                self.dim_dict[dim_symbol] = dim_power
        else:
            raise TypeError(("Dimension can be constructed with either a "
                             "string among {}, either None, either a "
                             "dictionnary with keys included in {}, "
                             "but not {}.").format(SI_SYMBOL_LIST,
                                                   SI_SYMBOL_LIST,
                                                   definition))

    def __str__(self):
        """Concatenate symbol-wise the content of the dim_dict attribute."""
        dim_symbol_list = self.dim_dict.items()
        output_init = 1
        output = output_init
        for (dim_symbol, dim_power) in dim_symbol_list:
            output *= sp.Symbol(dim_symbol)**dim_power
        if output == output_init:
            return NO_DIMENSION_STR
        else:
            return str(output)

    def __repr__(self):
        """Return the dim_dict into a <Dimension : ...> tag."""
        return "<Dimension : " + str(self.dim_dict) + ">"

    def __mul__(self, y):
        """Allow the multiplication of Dimension objects."""
        if isinstance(y, Dimension):
            new_dim_dict = {}
            for dim_symbol in self.dim_dict.keys():
                new_dim_dict[dim_symbol] = (self.dim_dict[dim_symbol]
                                            + y.dim_dict[dim_symbol])
            return Dimension(new_dim_dict)
        else:
            raise TypeError(("A dimension can only be multiplied "
                             "by another dimension, not {}.").format(y))

    __rmul__ = __mul__

    def __div__(self, y):
        """Allow the division of Dimension objects."""
        if isinstance(y, Dimension):
            new_dim_dict = {}
            for dim_symbol in self.dim_dict.keys():
                new_dim_dict[dim_symbol] = (self.dim_dict[dim_symbol]
                                            - y.dim_dict[dim_symbol])
            return Dimension(new_dim_dict)
        elif y == 1:  # allowing division by one
            return self
        else:
            raise TypeError(("A dimension can only be divided "
                             "by another dimension, not {}.").format(y))

    def __rdiv__(self, x):
        """Only used to raise a TypeError."""
        if x == 1:   # allowing one-divion
            return self.__inv__()
        else:
            raise TypeError("A Dimension can only divide 1 to be inverted.")

    __truediv__ = __div__

    __rtruediv__ = __rdiv__

    def __pow__(self, y):
        """Allow the elevation of Dimension objects to a real power."""
        if np.isscalar(y):
            new_dim_dict = {}
            for dim_symbol in self.dim_dict.keys():
                new_dim_dict[dim_symbol] = (self.dim_dict[dim_symbol]
                                            * y)
            return Dimension(new_dim_dict)
        else:
            raise TypeError(("The power of a dimension must be real,"
                             "not {}").format(type(y)))

    def __eq__(self, y):
        """Dimensions are equal if their dim_dict are equal."""
        return self.dim_dict == y.dim_dict

    def __ne__(self, y):
        """Return not (self == y)."""
        return not self.__eq__(y)

    def __inv__(self):
        """Inverse the dimension by taking the negative of the powers."""
        new_dim_dict = {key: -value for (key, value) in self.dim_dict.items()}
        return Dimension(new_dim_dict)

    def str_SI_unit(self):
        """Concatenate symbol-wise the unit."""
        dim_symbol_list = self.dim_dict.items()
        output_init = 1
        output = output_init
        for (dim_symbol, dim_power) in dim_symbol_list:
            output *= sp.Symbol(SI_UNIT_SYMBOL[dim_symbol])**dim_power
        if output == output_init:
            return ""
        else:
            return str(output)
