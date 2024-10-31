"""physipy : physical quantities in python

This is the __init__ docstring of physipy.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

from ._constants import constants, scipy_constants, scipy_constants_codata
from ._version import __version__
from .quantity import (
    A,
    Dimension,
    DimensionError,
    K,
    Quantity,
    add_back_unit_param,
    asqarray,
    cd,
    check_dimension,
    decorate_with_various_unit,
    dimension_and_favunit,
    dimensionify,
    drop_dimension,
    imperial_units,
    kg,
    m,
    make_quantity,
    mol,
    plotting_context,
    quantify,
    rad,
    s,
    set_favunit,
    setup_matplotlib,
    sr,
    units,
    utils,
)
