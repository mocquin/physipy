# !/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import pi

from ._version import __version__
from . import quantity
from .quantity import Quantity, Dimension, make_quantity, quantify, DimensionError, dimensionify
from .quantity import check_dimension, set_favunit, dimension_and_favunit, drop_dimension, decorate_with_various_unit, add_back_unit_param, array_to_Q_array
from .quantity import quad, dblquad, tplquad, root, brentq
from .quantity import vectorize
from .quantity import m, kg, s, A, K, cd, mol, rad, sr, SI_units, SI_units_prefixed, SI_derived_units, other_units, units

from .quantity import setup_matplotlib

from .constants import constants, scipy_constants, scipy_constants_codata
from .custom_units import custom_units, imperial_units
