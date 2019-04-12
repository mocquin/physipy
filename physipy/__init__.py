# !/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import pi

from ._version import __version__
from . import quantity
from .quantity import Quantity, Dimension, make_quantity, quantify, DimensionError
from .quantity import trapz, quad, dblquad, tplquad, qroot, qbrentq
from .quantity import linspace, interp, vectorize
from .quantity import m, kg, s, A, K, cd, mol, rad, sr, SI_units, SI_units_prefixed, SI_derived_units, other_units, units

from .constants import constants, scipy_constants, scipy_constants_codata
from .custom_units import custom_units, imperial_units
