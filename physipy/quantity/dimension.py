# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Manipulation of physical dimensions.

This module defines `Dimension`, which represents the physical dimensions of a
quantity as exponents over the SI base dimensions (length, mass, time, ...),
plus `DimensionError` and the string/sympy helpers used to parse and render
dimension expressions.

Note:
    ``rad`` (plane angle) and ``sr`` (solid angle) are not SI base dimensions
    but are included for convenience. The string representation relies on sympy,
    and numpy is used to check that dimension powers are scalars.
"""
# Developer notes (kept out of the rendered docs).
#
# PROPOSITIONS:
#  * method to return a latex-formated str ?
#  * change the str/repr style to a table-view of the dimension content ?
#  * should sr be just a unit with dimension rad**2 ?
#  * add a full-named repr ? (ex: "length/time")
#  * should Dimension implement add/sub operation (allowed when dims are equal) ?
#  * change the dimension representation from dict to array (faster) ?
#  * allow construction with strings (Dimension("m**2") or Dimension ("L**2")) ?
#  * could define a contains method to check if a dimension is not 0
#  * try to not relie on numpy/sympy
#  * should allow complex exponent ?
#  * move has_integer_dimension from Quantity to Dimension ?
#  * allow None definition with Dimension() ?
#  * implementation should not rely on the dimension system choosen
from __future__ import annotations

import json
import numbers
import os
from typing import Any, Literal, Union, Optional

import numpy as np

from .._optional import require

# sympy is now an *optional* dependency for this module : it is only needed to
# (B) parse compound dimension strings like "L**2/T" and (C) render LaTeX. The
# common str/repr path is pure-Python (see format_power_dict), so sympy is
# imported lazily, inside the few functions that still use it.

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "dimension.txt")) as file:
    SI_UNIT_SYMBOL = json.load(file)


Scalar = Union[int, float]


SI_SYMBOL_LIST = list(SI_UNIT_SYMBOL.keys())
NO_DIMENSION_STR = "no-dimension"

NULL_SI_DICT: dict[str, Scalar] = {dim: 0 for dim in SI_SYMBOL_LIST}


def parse_str_to_dic(exp_str: str) -> dict:
    """Parse a str expression into a power dict.

    Parameters
    ----------
    exp_str : str
        A string representing an expression of products with exponents.

    Returns
    -------
    exp_dic : dict
        A dict with keys the string symbol of the expression, and values the corresponding
        exponent.

    Examples
    --------
    >>> parse_str_to_dic("L**2/M")
    {'L': 2, 'M': -1}
    """
    # Parsing a compound string is the one construction path that needs sympy
    # (extra: "symbolic"). Building a Dimension from a dict or a single base
    # symbol does not go through here.
    sympy = require("sympy", "symbolic")
    parse_expr = require("sympy.parsing.sympy_parser", "symbolic").parse_expr
    parsed = parse_expr(
        exp_str,
        global_dict={"Symbol": sympy.Symbol, "Integer": sympy.Integer},
    )
    exp_dic = {
        str(key): value for key, value in parsed.as_powers_dict().items()
    }
    return exp_dic


def check_pattern(exp_str: str, symbol_list: list) -> bool:
    """Check that all symbols used in exp_str are present in symbol_list.

    Start by parsing the string expression into a power_dict, then check that
    all the keys of the power_dict are present in the symbol_list.

    Parameters
    ----------
    exp_str : str
        A string representing an expression of products with exponents.
    symbol_list : list of str
        A list of the base symbol that are allowed.

    Returns
    -------
    bool
        A bool indicating if all the symbol in the expression are allowed in the symbol
        list.

    Examples
    --------
    >>> check_pattern("L**2*T", ["L", "T"])
    True
    >>> check_pattern("L**2*M", ["L", "T"])
    False
    """
    exp_dic = parse_str_to_dic(exp_str)
    return set(exp_dic.keys()).issubset(set(symbol_list))


class DimensionError(Exception):
    """Exception class for dimension errors."""

    def __init__(
        self, dim_1: Dimension, dim_2: Dimension, binary: bool = True
    ) -> None:
        """Init method of DimensionError class."""
        if binary:
            self.message = (
                "Dimension error : dimensions of "
                f"operands are {dim_1} and {dim_2}, and are "
                f"differents ({dim_1.dimensionality} vs {dim_2.dimensionality})."
            )
        else:
            self.message = (
                f"Dimension error : dimension is {dim_1} "
                f"but should be {dim_2} ({dim_1.dimensionality} vs {dim_2.dimensionality})."
            )

    def __str__(self) -> str:
        """Str method of DimensionError class."""
        return self.message


class Dimension(object):
    """Allows to manipulate physical dimensions."""

    # DEFAULT REPR LATEX can be used to change the way a Dimension
    # object is displayed in JLab
    DEFAULT_REPR_LATEX = "dim_dict"  # "SI_unit"
    __slots__ = "dim_dict"
    dim_dict: dict[str, Scalar]

    def __init__(
        self,
        definition: Optional[Union[str, dict[str, Union[int, float]]]] = None,
    ) -> None:
        """Allow the creation of Dimension object with 3 possibile ways."""
        self.dim_dict = NULL_SI_DICT.copy()
        if definition is None:
            pass  # dim_dict already initialized
        # most of the time, the definition is a dim_dict of another quantity
        # so it already has the good shape
        elif isinstance(definition, dict) and set(
            list(definition.keys())
        ) == set(SI_SYMBOL_LIST):
            # copy to avoid aliasing the caller's dict into this (hashable,
            # treated-as-immutable) Dimension
            self.dim_dict = definition.copy()
        # example : {"L":1, "T":-2}
        elif isinstance(definition, dict) and set(
            list(definition.keys())
        ).issubset(
            SI_SYMBOL_LIST
        ):  # and
            # all([np.isscalar(v) for v in definition.values()])):
            for dim_symbol, dim_power in definition.items():
                self.dim_dict[dim_symbol] = dim_power
        # example : "L"
        elif isinstance(definition, str) and definition in list(
            self.dim_dict.keys()
        ):
            self.dim_dict[definition] = 1
        # example : "L**2/T**3"
        elif isinstance(definition, str) and check_pattern(
            definition, SI_UNIT_SYMBOL.keys()
        ):
            definition = parse_str_to_dic(definition)
            for dim_symbol, dim_power in definition.items():
                if dim_power == int(dim_power):
                    dim_power = int(dim_power)
                self.dim_dict[dim_symbol] = dim_power
        # example : "m"
        elif isinstance(definition, str) and check_pattern(
            definition, SI_UNIT_SYMBOL.values()
        ):
            definition = parse_str_to_dic(definition)
            for my_si_symbol, dim_power in definition.items():
                if dim_power == int(dim_power):
                    dim_power = int(dim_power)
                dim_symbol = [
                    dim_symbol
                    for dim_symbol, si_symbol in SI_UNIT_SYMBOL.items()
                    if my_si_symbol == si_symbol
                ][0]
                self.dim_dict[dim_symbol] = dim_power
        else:
            raise TypeError(
                (
                    "Dimension can be constructed with either a "
                    "string among {}, either None, either a "
                    "dictionnary with keys included in {}, "
                    "either a string of sympy expression with "
                    "those same keys "
                    "but not {}."
                ).format(SI_SYMBOL_LIST, SI_SYMBOL_LIST, definition)
            )

    def __str__(self: Dimension) -> str:
        """Concatenate symbol-wise the content of the dim_dict attribute."""
        return compute_str(self.dim_dict, NO_DIMENSION_STR)

    def __format__(self: Dimension, format_spec: str) -> str:
        raw = self.__str__()
        return format(raw, format_spec)

    def __repr__(self: Dimension) -> str:
        """Return the dim_dict into a <Dimension : ...> tag."""
        return "<Dimension : " + str(self.dim_dict) + ">"

    def _repr_latex_(self: Dimension):
        """Latex repr hook for IPython.

        LaTeX rendering is an optional feature (extra: "symbolic") : without
        sympy, returning None tells IPython to fall back to the plain-text
        repr instead of raising.
        """
        try:
            latex = require("sympy.printing", "symbolic").latex
        except ImportError:
            return None
        if self.DEFAULT_REPR_LATEX == "dim_dict":
            expr_dim = expand_dict_to_expr(self.dim_dict)
            return "$" + str(latex(expr_dim)) + "$"
        else:  # self.DEFAULT_REPR_LATEX == "SI_unit":
            return self.latex_SI_unit()

    def __mul__(self: Dimension, y: Dimension) -> Dimension:
        """Multiply Dimension objects.

        The multiplication of 2 Dimension objects is another Dimension with power given by
        the sum of the input Dimension's powers.

        Parameter
        ---------
        y : Dimension
            The Dimension to multiply.

        Returns
        -------
        dim : Dimension
            The new Dimension representing the product.
        """
        # if isinstance(y, Dimension):
        try:
            new_dim_dict = {
                d: self.dim_dict[d] + y.dim_dict[d]
                for d in self.dim_dict.keys()
            }
            return Dimension(new_dim_dict)
        except Exception as e:
            raise TypeError(
                (
                    "A dimension can only be multiplied "
                    "by another dimension, not {}."
                    "Got exception {}"
                ).format(y, e)
            )

    __rmul__ = __mul__

    def __truediv__(self: Dimension, y: Dimension) -> Dimension:
        """Allow the division of Dimension objects.

        The division of 2 Dimension objects is another Dimension with power given by
        the difference between the input Dimension's powers.

        Parameter
        ---------
        y : Dimension
            The Dimension to divide by.

        Returns
        -------
        dim : Dimension
            The new Dimension representing the division.
        """
        # if isinstance(y, Dimension):
        try:
            new_dim_dict = {
                d: self.dim_dict[d] - y.dim_dict[d]
                for d in self.dim_dict.keys()
            }
            return Dimension(new_dim_dict)
        # elif y == 1:  # allowing division by one
        #    return self
        # else:
        except Exception as e:
            raise TypeError(
                (
                    "A dimension can only be divided "
                    "by another dimension, not {}."
                    "Got exception {}"
                ).format(y, e)
            )

    def __rtruediv__(self: Dimension, x: Dimension) -> Dimension:
        """Inverse a Dimension by divinding one.

        THe only value a Dimension can divide is 1, in order to invert a Dimension.
        The inverse of the Dimension has all it's power negated.
        Mainly used to raise a TypeError in unallowed uses.

        Parameter
        ---------
        x : 1
            A Dimension can only divide one.

        Returns
        -------
        dim : Dimension
            The inverse of the input Dimension.

        """
        if x == 1:  # allowing one-divion
            # return self.inverse()
            return self**-1
        else:
            raise TypeError("A Dimension can only divide 1 to be inverted.")

    def __pow__(self: Dimension, y: Scalar) -> Dimension:
        """Raise a Dimension objects to a real power.

        Only scalars are allowed.

        Parameter
        ---------
        y : scalar-like
            The exponent.

        Returns
        -------
        dim : Dimension
            The raised Dimension.
        """
        if isinstance(y, numbers.Number):  # np.isscalar(y):
            new_dim_dict = {
                d: self.dim_dict[d] * y for d in self.dim_dict.keys()
            }
            return Dimension(new_dim_dict)
        else:
            raise TypeError(
                ("The power of a dimension must be a scalar," f"not {type(y)}")
            )

    def __eq__(self, y: object) -> bool:
        """Check equality between Dimension objects.

        Dimensions are equal if their dim_dict are equal.

        Parameter
        ---------
        y : Dimension
            The Dimension object to test equality with.
        Returns
        -------
        bool
            True if the Dimensions objects are equal, False otherwise.
        """
        if isinstance(y, Dimension):
            return self.dim_dict == y.dim_dict
        return False

    def __hash__(self) -> int:
        return hash(str(self))

    # def __ne__(self, y):
    #    """Return not (self == y)."""
    #    return not self.__eq__(y)

    # def inverse(self):
    #    """Inverse the dimension by taking the negative of the powers."""
    #    inv_dict = {key: -value for key, value in self.dim_dict.items()}
    #    return Dimension(inv_dict)

    def siunit_dict(self: Dimension) -> dict:
        """Return a dict where keys are SI unit string, and value are powers.

        Returns
        -------
        dict
            A dict with keys the SI-unit symbols and values the corresponding exponent.
        """
        return {
            SI_UNIT_SYMBOL[key]: value for key, value in self.dim_dict.items()
        }

    def str_SI_unit(self: Dimension) -> str:
        """Compute the symbol-wise SI unit equivalent of the Dimension.

        Returns
        -------
        str
            A string that represent the Dimension powers with SI-units symbols.

        See also
        --------
        compute_str, Dimension.siunit_dict
        """
        str_dict = self.siunit_dict()
        return compute_str(str_dict, "")

    def latex_SI_unit(self: Dimension) -> str:
        """Latex repr of SI unit form.

        Leverage sympy's latex function to compute the latex expression equivalent to
        the SI-unit string representation of the Dimension.

        See also
        --------
        Dimension.siunit_dict
        """
        # Explicit LaTeX request : require sympy (extra: "symbolic").
        latex = require("sympy.printing", "symbolic").latex
        expr_SI = expand_dict_to_expr(self.siunit_dict())
        return "$" + str(latex(expr_SI)) + "$"

    @property
    def dimensionality(self: Dimension) -> str:
        """Return the first dimensionality with same dimension found in DIMENSIONALITY.

        Returns
        -------
        str
            A string giving a dimensionnality equivalent to the Dimension.

        See also
        --------
        DIMENSIONALITY
        """
        try:
            return [
                dimensionality
                for dimensionality, dimension in DIMENSIONALITY.items()
                if dimension == self
            ][0]
        except BaseException:
            return str(self)


DIMENSIONLESS: Dimension = Dimension(None)


def _format_exponent(exp: Scalar) -> str:
    """Render an exponent exactly like sympy's str printer does."""
    return str(exp)


def _format_factor(name: str, exp: Scalar) -> str:
    """Render a single ``name**exp`` factor with a *positive* exponent.

    Used for both numerator factors and the absolute exponents of denominator
    factors : an integer exponent of 1 collapses to the bare symbol.
    """
    if isinstance(exp, int) and exp == 1:
        return name
    return f"{name}**{_format_exponent(exp)}"


def format_power_dict(power_dict: dict, default_str: str) -> str:
    """Convert a power-dict to its canonical string, without sympy.

    Pure-Python replacement for the sympy-backed rendering that used to power
    ``str``/``repr`` of :class:`Dimension`. It reproduces sympy's string
    printer byte-for-byte for the monomials physipy builds (products and
    quotients of ``symbol**exponent``), so the displayed form is unchanged :

      - symbols are sorted by name ;
      - positive exponents form the numerator, negative *integer* exponents
        form the denominator (``a*b**2/(c*d**3)``) ;
      - a lone factor follows sympy's special cases : ``x`` (exp 1),
        ``1/x`` (exp -1), ``x**(-2)`` / ``x**0.5`` otherwise.

    sympy is still used for the LaTeX rendering (see
    :func:`expand_dict_to_expr`), this only removes it from the common
    ``str``/``repr`` path.

    Parameters
    ----------
    power_dict : dict
        A power-dict mapping symbol strings to scalar exponents.
    default_str : str
        Returned when every exponent is zero (the "dimensionless" marker).

    See also
    --------
    expand_dict_to_expr : the sympy-based expression, kept for LaTeX output.
    """
    items = [(key, exp) for key, exp in power_dict.items() if exp != 0]
    if not items:
        return default_str
    items.sort(key=lambda kv: kv[0])

    # A lone factor : sympy keeps these as a single Pow, with special cases.
    if len(items) == 1:
        key, exp = items[0]
        if isinstance(exp, int) and exp == 1:
            return key
        if isinstance(exp, int) and exp == -1:
            return f"1/{key}"
        rendered = _format_exponent(exp)
        return f"{key}**({rendered})" if exp < 0 else f"{key}**{rendered}"

    # Two or more factors : sympy prints them as a numerator/denominator Mul.
    numerator = [(key, exp) for key, exp in items if exp > 0]
    denominator = [(key, -exp) for key, exp in items if exp < 0]
    num_str = (
        "*".join(_format_factor(k, e) for k, e in numerator)
        if numerator
        else "1"
    )
    if not denominator:
        return num_str
    den_factors = [_format_factor(k, e) for k, e in denominator]
    den_str = (
        den_factors[0]
        if len(den_factors) == 1
        else "(" + "*".join(den_factors) + ")"
    )
    return f"{num_str}/{den_str}"


def compute_str(
    power_dict: dict, default_str: str, output_init: int = 1
) -> str:
    """Convert power-dict to a string expression equivalent.

    Compute the product-concatenation of the
    dict as key**value into a string.
    Only used for 'str' and 'repr' methods.

    Parameters
    ----------
    power_dict : dict
        A power-dict containing str as keys and scalars as values.
    default_str : str
        A string to return if the output value is equal to the output initial value.
    output_init : scalar-like, defaults to 1.
        Unused, kept for backward compatibility.

    Returns
    -------
    str
        The canonical string for the power-dict, or default_str if all the
        exponents are zero.

    See also
    --------
    format_power_dict : the pure-Python renderer doing the actual work.
    """
    return format_power_dict(power_dict, default_str)


def expand_dict_to_expr(power_dict: dict, output_init: int = 1) -> Any:
    """
    Compute the sympy expression from exponent dict, starting the product with ouptput=1.
    Used only for the (optional) LaTeX rendering of Dimension.

    Parameters
    ----------
    power_dict : dict
        A power-dict containing str as keys and scalars as values.
    output_init : scalar-like, default to 1.
        The initial value used to compute the value of the power-expression.

    Returns
    -------
    expr
        A sympy-expression equivalent to the power-dict.

    See also
    --------
    compute_str : Convert a power-dict to a str equivalent.
    """
    # LaTeX rendering is optional (extra: "symbolic").
    Symbol = require("sympy", "symbolic").Symbol
    output = output_init
    for key, value in power_dict.items():
        output *= Symbol(key) ** value
    return output


DIMENSIONALITY = {
    "dimensionless": Dimension(None),
    # Base dimension
    "length": Dimension("L"),
    "mass": Dimension("M"),
    "time": Dimension("T"),
    "electric_current": Dimension("I"),
    "temperature": Dimension("theta"),
    "amount_of_substance": Dimension("N"),
    "luminous_intensity": Dimension("J"),
    "plane_angle": Dimension("RAD"),
    "solid_angle": Dimension("SR"),
    #
    "area": Dimension({"L": 2}),
    "volume": Dimension({"L": 3}),
    "speed": Dimension({"L": 1, "T": -1}),
    "acceleration": Dimension({"L": 1, "T": -2}),
    "force": Dimension({"M": 1, "L": 1, "T": -2}),
    "energy": Dimension({"M": 1, "L": 2, "T": -2}),
    "power": Dimension({"M": 1, "L": 2, "T": -3}),
    "capacitance": Dimension({"M": -1, "L": -2, "T": 4, "I": 2}),
    "voltage": Dimension({"M": 1, "L": 2, "T": -3, "I": -1}),
}
