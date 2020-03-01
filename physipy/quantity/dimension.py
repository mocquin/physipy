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
 * [X] : try a cleaner conversion from dict to str
 * [X] : try to make SI_SYMBOL_LIST a list
 * [X] : move base dimension dict to a file

PROPOSITIONS:
 * method to return a latex-formated str ?
 * change the str/repr style to a table-view of the dimension content ?
 * should sr be just a unit with dimension rad**2 ?
 * add a full-named repr ? (ex: "length/time")
 * should Dimension implement add/sub operation (allowed when dims are equal) ?
 * change the dimension representation from dict to array (faster) ?
 * allow construction with strings (Dimension("m**2") or Dimension ("L**2")) ?
 * could define a contains method to check if a dimension is not 0
 * try to not relie on numpy/sympy
 * should allow complex exponent ?
 * move has_integer_dimension from Quantity to Dimension ?

PLEASE NOTE :
- rad and sr are not base SI-units, but were added for convenience. They can be
    deleted if not needed, but update tests in consequence.
- this modules relies on :
 - sympy to compute the concatenated representation of the Dimension object
 - numpy to check if the dimension powers are scalars

"""
import json
import os

import sympy as sp
import numpy as np


dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "dimension.txt")) as file:
    SI_UNIT_SYMBOL = json.load(file)


SI_SYMBOL_LIST = list(SI_UNIT_SYMBOL.keys())
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
        self.dim_dict = {dim: 0 for dim in SI_SYMBOL_LIST}
        if definition is None:
            pass  # dim_dict already initialized
        elif definition in list(self.dim_dict.keys()):
            self.dim_dict[definition] = 1
        elif (isinstance(definition, dict) and
              set(list(definition.keys())).issubset(SI_SYMBOL_LIST) and
             all([np.isscalar(v) for v in definition.values()])):
            for dim_symbol, dim_power in definition.items():
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
        return compute_str(self.dim_dict, NO_DIMENSION_STR)

    def __repr__(self):
        """Return the dim_dict into a <Dimension : ...> tag."""
        return "<Dimension : " + str(self.dim_dict) + ">"

    def __mul__(self, y):
        """Allow the multiplication of Dimension objects."""
        if isinstance(y, Dimension):
            new_dim_dict = {d: self.dim_dict[d] + y.dim_dict[d] for d in self.dim_dict.keys()}
            return Dimension(new_dim_dict)
        else:
            raise TypeError(("A dimension can only be multiplied "
                             "by another dimension, not {}.").format(y))

    __rmul__ = __mul__

    def __truediv__(self, y):
        """Allow the division of Dimension objects."""
        if isinstance(y, Dimension):
            new_dim_dict = {d: self.dim_dict[d] - y.dim_dict[d] for d in self.dim_dict.keys()}
            return Dimension(new_dim_dict)
        #elif y == 1:  # allowing division by one
        #    return self
        else:
            raise TypeError(("A dimension can only be divided "
                             "by another dimension, not {}.").format(y))

    def __rtruediv__(self, x):
        """Only used to raise a TypeError."""
        if x == 1:   # allowing one-divion
            return self.inverse()
        else:
            raise TypeError("A Dimension can only divide 1 to be inverted.")

    def __pow__(self, y):
        """Allow the elevation of Dimension objects to a real power."""
        if np.isscalar(y):
            new_dim_dict = {d: self.dim_dict[d] * y for d in self.dim_dict.keys()}
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

    def inverse(self):
        """Inverse the dimension by taking the negative of the powers."""
        inv_dict = {key: -value for key, value in self.dim_dict.items()}
        return Dimension(inv_dict)

    def str_SI_unit(self):
        """Compute the symbol-wise SI unit."""
        str_dict = {SI_UNIT_SYMBOL[key]: value for key, value in self.dim_dict.items()}
        return compute_str(str_dict, "")


def compute_str(dic, default):
    """Compute the product-concatenation of the dict as key**value."""
    output_init = 1
    output = output_init
    for key, value in dic.items():
        output *= sp.Symbol(key)**value
    if output == output_init:
        return default
    else:
        return str(output)
