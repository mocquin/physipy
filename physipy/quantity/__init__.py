# !/usr/bin/env python
# -*- coding: utf-8 -*-

from ._plot import plotting_context, setup_matplotlib
from ._units import A, K, cd, imperial_units, kg, m, mol, rad, s, sr, units
from .quantity import (
    SI_UNIT_SYMBOL,
    Dimension,
    DimensionError,
    Quantity,
    dimensionify,
    make_quantity,
    quantify,
)
from .utils import (
    add_back_unit_param,
    asqarray,
    check_dimension,
    decorate_with_various_unit,
    dimension_and_favunit,
    drop_dimension,
    set_favunit,
)
