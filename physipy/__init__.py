# !/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import pi

from ._version import __version__
from . import quantity
from .quantity import Quantity, Dimension, make_quantity, quantify, DimensionError, dimensionify
from .quantity import check_dimension, set_favunit, dimension_and_favunit, drop_dimension, decorate_with_various_unit, add_back_unit_param, asqarray
from .integrate import quad, dblquad, tplquad
from .optimize import root, brentq
from .quantity import m, kg, s, A, K, cd, mol, rad, sr
from .quantity import SI_units, SI_units_prefixed, SI_derived_units, other_units, units, all_units, SI_derived_units_prefixed

from .quantity import setup_matplotlib

from .constants import constants, scipy_constants, scipy_constants_codata
from .custom_units import custom_units, imperial_units

from . import math


try:
    import uncertainties as uc
    from .quantity.quantity import register_property_backend

    uncertainties_property_backend_interface = {
        # res is the backend result of the attribute lookup, and q the wrapping quantity
        "nominal_value":lambda q, res: q._SI_unitary_quantity*res,
        "std_dev":lambda q, res: q._SI_unitary_quantity*res,
        "n":lambda q, res: q._SI_unitary_quantity*res,
        "s":lambda q, res: q._SI_unitary_quantity*res,
    }

    register_property_backend(uc.core.Variable, 
                             uncertainties_property_backend_interface)
except:
    pass