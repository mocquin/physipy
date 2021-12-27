# !/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
import matplotlib.units as munits

from .quantity import Quantity, Dimension, quantify, make_quantity, DimensionError
from .units import all_units

class QuantityConverter(munits.ConversionInterface):

    @staticmethod
    def default_units(q, axis):
        if axis.units is not None:
            if not q.dimension == axis.units.dimension:
                raise DimensionError(q.dimension, axis.units.dimension)
        if isinstance(q, Quantity):
            q_unit = q._plot_extract_q_for_axe(all_units.values())
            return q_unit
        # when calling ax.set_xlim(2*m, 3*m), the tuple (2*m, 3*m) is passed to
        # default_units 
        if np.iterable(q):
            for v in q:
                if isinstance(v, Quantity):
                    return v._plot_extract_q_for_axe(all_units.values())
            return None

    @staticmethod
    def axisinfo(q_unit, axis):
        if axis.units is not None:
            if not q_unit.dimension == axis.units.dimension:
                raise DimensionError(q_unit.dimension, axis.units.dimension)
        return munits.AxisInfo(label='{}'.format(q_unit.symbol))

    def convert(self, q, q_unit, axis):
        if isinstance(q, (tuple, list)):
            return [self._convert(v, q_unit, axis) for v in q]
        else:
            return self._convert(q, q_unit, axis)
        
    def _convert(self, q, q_unit, axis):
        if not isinstance(q_unit, Quantity):
            raise TypeError(f"Expect Quantity for q_unit, but got {type(q_unit)} for {q_unit}")
        return q._plot_get_value_for_plot(q_unit)    
    
def setup_matplotlib(enable=True):
    if matplotlib.__version__ < '2.0':
        raise RuntimeError('Matplotlib >= 2.0 required to work with units.')
    if enable == False:
        munits.registry.pop(Quantity, None)
    else:
        munits.registry[Quantity] = QuantityConverter()
        
def plotting_context():
    """Context for plotting with Quantity objects
    Based on : 
        https://docs.astropy.org/en/stable/_modules/astropy/visualization/units.html#quantity_support
    """
    
    from matplotlib import units
    from matplotlib import ticker

    # Get all subclass for Quantity, since matplotlib checks on class,
    # not subclass.
    def all_issubclass(cls):
        return {cls}.union(
            [s for c in cls.__subclasses__() for s in all_issubclass(c)])


    class MplQuantityConverter(QuantityConverter):

        _all_issubclass_quantity = all_issubclass(Quantity)

        def __init__(self):

            # Keep track of original converter in case the context manager is
            # used in a nested way.
            self._original_converter = {}

            for cls in self._all_issubclass_quantity:
                self._original_converter[cls] = munits.registry.get(cls)
                munits.registry[cls] = self

        def __enter__(self):
            return self

        def __exit__(self, type, value, tb):
            for cls in self._all_issubclass_quantity:
                if self._original_converter[cls] is None:
                    del munits.registry[cls]
                else:
                    munits.registry[cls] = self._original_converter[cls]

    return MplQuantityConverter()