# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""allows manipulating physical Dimension objects.

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
 * allow None definition with Dimension() ?
 * implementation should not rely on the dimension system choosen

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
from sympy.parsing.sympy_parser import parse_expr
import sympy.printing.latex as latex


dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "dimension.txt")) as file:
    SI_UNIT_SYMBOL = json.load(file)


SI_SYMBOL_LIST = list(SI_UNIT_SYMBOL.keys())
NO_DIMENSION_STR = "no-dimension"

NULL_SI_DICT = {dim: 0 for dim in SI_SYMBOL_LIST}


def parse_str_to_dic(exp_str):
    """Parse a str expression into a power dict.
    
    Example:
    ========
    >>> parse_str_to_dic("L**2/M")
    {"L":2, "M":1}
    """
    parsed = parse_expr(exp_str, global_dict={'Symbol':sp.Symbol, 'Integer':sp.Integer})
    exp_dic = {str(key):value for key,value in parsed.as_powers_dict().items()}
    return exp_dic


def check_pattern(exp_str, symbol_list):
    """Check that all symbols used in exp_str are present in symbol_list.
    
    Start by parsing the string expression into a power_dict, then check that 
    all the keys of the power_dict are present in the symbol_list.
    
    Example:
    ========
    >>> check_pattern("L**2*T", ["L", "T"])
    True
    >>> check_pattern("L**2*M", ["L", "T"])
    False
    """
    exp_dic = parse_str_to_dic(exp_str)
    return set(exp_dic.keys()).issubset(set(symbol_list))


class DimensionError(Exception):
    """Exception class for dimension errors."""

    def __init__(self, dim_1, dim_2, binary=True):
        """Init method of DimensionError class."""
        if binary:
            self.message = ("Dimension error : dimensions of "
                            f"operands are {dim_1} and {dim_2}, and are "
                            f"differents ({dim_1.dimensionality} vs {dim_2.dimensionality}).")
        else:
            self.message = (f"Dimension error : dimension is {dim_1} "
                            f"but should be {dim_2} ({dim_1.dimensionality} vs {dim_2.dimensionality}).")

    def __str__(self):
        """Str method of DimensionError class."""
        return self.message


class Dimension(object):
    """Allows to manipulate physical dimensions."""
    
    # DEFAULT REPR LATEX can be used to change the way a Dimension
    # object is displayed in JLab
    DEFAULT_REPR_LATEX = "dim_dict" # "SI_unit"

    def __init__(self, definition):
        """Allow the creation of Dimension object with 3 possibile ways."""
        self.dim_dict = NULL_SI_DICT.copy()
        if definition is None:
            pass  # dim_dict already initialized
        # most of the time, the definition is a dim_dict of another quantity
        # so it already has the good shape
        elif (isinstance(definition, dict) and 
            set(list(definition.keys()))==set(SI_SYMBOL_LIST)): 
                self.dim_dict = definition
        # example : {"L":1, "T":-2}
        elif (isinstance(definition, dict) and
             set(list(definition.keys())).issubset(SI_SYMBOL_LIST)): #and
             #all([np.isscalar(v) for v in definition.values()])):
            for dim_symbol, dim_power in definition.items():
                self.dim_dict[dim_symbol] = dim_power
        # example : "L"
        elif definition in list(self.dim_dict.keys()):
            self.dim_dict[definition] = 1
        # example : "L**2/T**3"
        elif (isinstance(definition, str) and check_pattern(definition, SI_UNIT_SYMBOL.keys())):
            definition = parse_str_to_dic(definition)
            for dim_symbol, dim_power in definition.items():
                if dim_power == int(dim_power):
                    dim_power = int(dim_power)
                self.dim_dict[dim_symbol] = dim_power
        # example : "m"
        elif (isinstance(definition, str) and check_pattern(definition, SI_UNIT_SYMBOL.values())):
            definition = parse_str_to_dic(definition)
            for my_si_symbol, dim_power in definition.items():
                if dim_power == int(dim_power):
                    dim_power = int(dim_power)
                dim_symbol = [dim_symbol for dim_symbol, si_symbol in SI_UNIT_SYMBOL.items() if my_si_symbol == si_symbol][0]
                self.dim_dict[dim_symbol] = dim_power
        else:
            raise TypeError(("Dimension can be constructed with either a "
                             "string among {}, either None, either a "
                             "dictionnary with keys included in {}, "
                             "either a string of sympy expression with "
                             "those same keys "
                             "but not {}.").format(SI_SYMBOL_LIST,
                                                   SI_SYMBOL_LIST,
                                                   definition))

    def __str__(self):
        """Concatenate symbol-wise the content of the dim_dict attribute."""
        return compute_str(self.dim_dict, NO_DIMENSION_STR)

    def __format__(self, format_spec):
        raw = self.__str__()
        return format(raw, format_spec)
    
    def __repr__(self):
        """Return the dim_dict into a <Dimension : ...> tag."""
        return "<Dimension : " + str(self.dim_dict) + ">"

    def _repr_latex_(self):
        """Latex repr hook for IPython."""
        if self.DEFAULT_REPR_LATEX == "dim_dict":
            expr_dim = expand_dict_to_expr(self.dim_dict)
            return "$" + latex(expr_dim) + "$"
        else:# self.DEFAULT_REPR_LATEX == "SI_unit":
            return self.latex_SI_unit()

    
    def __mul__(self, y):
        """Allow the multiplication of Dimension objects."""
        #if isinstance(y, Dimension):
        try:
            new_dim_dict = {d: self.dim_dict[d] + y.dim_dict[d] for d in self.dim_dict.keys()}
            return Dimension(new_dim_dict)
        except Exception as e:
            raise TypeError(("A dimension can only be multiplied "
                             "by another dimension, not {}. Got exception {}").format(y, e))

    __rmul__ = __mul__

    def __truediv__(self, y):
        """Allow the division of Dimension objects."""
        #if isinstance(y, Dimension):
        try:
            new_dim_dict = {d: self.dim_dict[d] - y.dim_dict[d] for d in self.dim_dict.keys()}
            return Dimension(new_dim_dict)
        #elif y == 1:  # allowing division by one
        #    return self
       # else:
        except Exception as e:
            raise TypeError(("A dimension can only be divided "
                             "by another dimension, not {}. Got exception {}").format(y, e))

    def __rtruediv__(self, x):
        """Only used to raise a TypeError."""
        if x == 1:   # allowing one-divion
            #return self.inverse()
            return self**-1
        else:
            raise TypeError("A Dimension can only divide 1 to be inverted.")

    def __pow__(self, y):
        """Allow the elevation of Dimension objects to a real power."""
        if np.isscalar(y):
            new_dim_dict = {d: self.dim_dict[d] * y for d in self.dim_dict.keys()}
            return Dimension(new_dim_dict)
        else:
            raise TypeError(("The power of a dimension must be a scalar,"
                             "not {}").format(type(y)))

    def __eq__(self, y):
        """Dimensions are equal if their dim_dict are equal."""
        #if type(y)==Dimension:
        try:
            return self.dim_dict == y.dim_dict
        #else:
        except:
            return False
    #def __ne__(self, y):
    #    """Return not (self == y)."""
    #    return not self.__eq__(y)

    #def inverse(self):
    #    """Inverse the dimension by taking the negative of the powers."""
    #    inv_dict = {key: -value for key, value in self.dim_dict.items()}
    #    return Dimension(inv_dict)

    def siunit_dict(self):
        """Return a dict where keys are SI unit
        string, and value are powers."""
        return {SI_UNIT_SYMBOL[key]: value for key, value in self.dim_dict.items()}
    
    def str_SI_unit(self):
        """Compute the symbol-wise SI unit."""
        str_dict = self.siunit_dict()
        return compute_str(str_dict, "")

    def latex_SI_unit(self):
        """Latex repr of SI unit form."""
        expr_SI = expand_dict_to_expr(self.siunit_dict())
        return "$" + latex(expr_SI) + "$"
    
    @property
    def dimensionality(self):
        """Return the first dimensionality with same 
        dimension found in DIMENSIONALITY"""
        try:
            return [dimensionality for dimensionality, dimension in DIMENSIONALITY.items() if dimension == self][0]
        except:
            return str(self)

    
DIMENSIONLESS = Dimension(None)

def compute_str(dic, default_str, output_init=1):
    """Compute the product-concatenation of the 
    dict as key**value into a string.
    Only used for 'str' and 'repr' methods.
    """
    output = expand_dict_to_expr(dic, output_init)
    if output == output_init:
        return default_str
    else:
        return str(output)

    
def expand_dict_to_expr(dic, output_init=1):
    """Compute the sympy expression from exponent dict,
    starting the product with ouptput=1.
    Only used for 'str' and 'repr' methods.
    """
    output = output_init
    for key, value in dic.items():
        output *= sp.Symbol(key)**value
    return output


DIMENSIONALITY = {
    "dimensionless":      Dimension(None),
    # Base dimension
    "length":             Dimension("L"),
    "mass":               Dimension("M"),
    "time":               Dimension("T"),
    "electric_current":   Dimension("I"),
    "temperature":        Dimension("theta"),
    "amount_of_substance":Dimension("N"),
    "luminous_intensity": Dimension("J"),
    "plane_angle":        Dimension("RAD"),
    "solid_angle":        Dimension("SR"),
    
    #
    "area":               Dimension({"L":2}),
    "volume":             Dimension({"L":3}),
    
    "speed":              Dimension({"L":1, "T":-1}),
    "acceleration":       Dimension({"L":1, "T":-2}),
    "force":              Dimension({"M":1, "L":1, "T":-2}),
    "energy":             Dimension({"M":1, "L":2, "T":-2}),
    "power":              Dimension({"M":1, "L":2, "T":-3}),
    "capacitance":        Dimension({"M":-1, "L":-2, "T":4, "I":2}),
    "voltage":            Dimension({"M":1, "L":2, "T":-3, "I":-1}),
}