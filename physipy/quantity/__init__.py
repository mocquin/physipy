# !/usr/bin/env python
# -*- coding: utf-8 -*-

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


# matplotlib integration is an optional feature : resolve it lazily so that
# `import physipy.quantity` does not pull in matplotlib.
def __getattr__(name):
    if name in ("setup_matplotlib", "plotting_context"):
        from . import _plot

        return getattr(_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
