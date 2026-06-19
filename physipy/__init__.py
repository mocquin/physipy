"""physipy : physical quantities in python

This is the __init__ docstring of physipy.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

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
    quantify,
    rad,
    s,
    set_favunit,
    sr,
    units,
    utils,
)


# Optional-dependency-backed features are resolved lazily so that
# `import physipy` only needs numpy (+ sympy) : scipy is pulled in on first
# access of the constants, matplotlib on first access of the plot helpers.
def __getattr__(name):
    if name in ("constants", "scipy_constants", "scipy_constants_codata"):
        from . import _constants

        return getattr(_constants, name)
    if name in ("setup_matplotlib", "plotting_context"):
        from .quantity import _plot

        return getattr(_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
