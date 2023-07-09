"""physipy : physical quantities in python

This is the __init__ docstring of physipy.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

from ._version import __version__

from .quantity import Quantity, Dimension, make_quantity, quantify, DimensionError, dimensionify
from .quantity import check_dimension, set_favunit, dimension_and_favunit, drop_dimension, decorate_with_various_unit, add_back_unit_param, asqarray

from .quantity import setup_matplotlib, plotting_context
from .quantity import utils

from .quantity import m, kg, s, A, K, cd, mol, rad, sr, units, imperial_units
from ._constants import constants, scipy_constants, scipy_constants_codata
