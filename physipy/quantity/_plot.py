# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .._optional import require

# matplotlib powers this whole module (the converter subclasses
# matplotlib.units.ConversionInterface) ; it is imported lazily by
# physipy.__getattr__, so a clear error is raised only on first use.
matplotlib = require("matplotlib", "plotting")
munits = require("matplotlib.units", "plotting")
mticker = require("matplotlib.ticker", "plotting")

from ._units import imperial_units, units
from .quantity import (
    Dimension,
    DimensionError,
    Quantity,
    make_quantity,
    quantify,
)

# import registry as used trouhout, faster
munits_registry = munits.registry

all_units = {**units, **imperial_units}


class QuantityFormatter(mticker.ScalarFormatter):
    """Tick formatter that appends the axis unit symbol to each tick label.

    By the time matplotlib calls a formatter, ``convert`` has already
    divided the data by the axis unit, so the tick *values* are plain
    numbers expressed in that unit. We therefore only format the scalar
    with the inherited :class:`~matplotlib.ticker.ScalarFormatter` and tack
    the symbol on, e.g. ``1``, ``2``, ``3`` -> ``1 km``, ``2 km``, ``3 km``.
    """

    def __init__(self, symbol, **kwargs):
        super().__init__(**kwargs)
        self._symbol = symbol

    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        # ScalarFormatter returns "" for ticks it wants to hide ; keep those
        # empty rather than emitting a lone unit symbol.
        if not s:
            return s
        return f"{s} {self._symbol}"


class QuantityConverter(munits.ConversionInterface):
    #: When True, ``axisinfo`` attaches a :class:`QuantityFormatter` so the
    #: unit symbol is repeated on every tick. Default False keeps the unit on
    #: the axis label only (matplotlib/pint/astropy default behaviour).
    tick_labels_with_unit = False

    @staticmethod
    def default_units(q, axis):
        # q may be a single Quantity, or an iterable of them : when calling
        # ax.set_xlim(2*m, 3*m), the tuple (2*m, 3*m) is passed here. Resolve
        # the representative Quantity before checking dimensions, so a tuple
        # against an axis that already has units still raises a DimensionError
        # rather than an AttributeError.
        if isinstance(q, Quantity):
            representative = q
        elif np.iterable(q):
            representative = next(
                (v for v in q if isinstance(v, Quantity)), None
            )
        else:
            representative = None

        if representative is None:
            return None

        if axis.units is not None:
            if not representative.dimension == axis.units.dimension:
                raise DimensionError(
                    representative.dimension, axis.units.dimension
                )
        return representative._plot_extract_q_for_axe(all_units.values())

    def axisinfo(self, q_unit, axis):
        if axis.units is not None:
            if not q_unit.dimension == axis.units.dimension:
                raise DimensionError(q_unit.dimension, axis.units.dimension)
        symbol = "{}".format(q_unit.symbol)
        if self.tick_labels_with_unit:
            return munits.AxisInfo(
                majfmt=QuantityFormatter(symbol), label=symbol
            )
        return munits.AxisInfo(label=symbol)

    def convert(self, q, q_unit, axis):
        if isinstance(q, (tuple, list)):
            return [self._convert(v, q_unit, axis) for v in q]
        else:
            return self._convert(q, q_unit, axis)

    def _convert(self, q, q_unit, axis):
        from ..quantity import asqarray

        if not isinstance(q_unit, Quantity):
            raise TypeError(
                f"Expect Quantity for q_unit, but got {type(q_unit)} for {q_unit}"
            )
        return asqarray(q)._plot_get_value_for_plot(q_unit)


def setup_matplotlib(
    enable: bool = True, tick_labels_with_unit: bool = False
) -> None:
    """Enable unit system in Matplotlib for Quantity objects.

    Parameters
    ----------
    enable : bool
        Weither to activate the handling of physipy in matplotlib.
    tick_labels_with_unit : bool
        When True, repeat the unit symbol on every tick label (via
        :class:`QuantityFormatter`) in addition to the axis label. Default
        False keeps the unit on the axis label only.

    Returns
    -------
    None

    """
    if matplotlib.__version__ < "2.0":
        raise RuntimeError("Matplotlib >= 2.0 required to work with units.")
    if not enable:
        munits_registry.pop(Quantity, None)
    else:
        converter = QuantityConverter()
        converter.tick_labels_with_unit = tick_labels_with_unit
        munits_registry[Quantity] = converter


def plotting_context(tick_labels_with_unit: bool = False):
    """Context for plotting with Quantity objects
    Based on :
        https://docs.astropy.org/en/stable/_modules/astropy/visualization/units.html#quantity_support

    Parameters
    ----------
    tick_labels_with_unit : bool
        When True, repeat the unit symbol on every tick label (via
        :class:`QuantityFormatter`) in addition to the axis label.
    """

    from matplotlib import ticker, units

    # Get all subclass for Quantity, since matplotlib checks on class,
    # not subclass.
    def all_issubclass(cls):
        return {cls}.union(
            [s for c in cls.__subclasses__() for s in all_issubclass(c)]
        )

    class MplQuantityConverter(QuantityConverter):
        def __init__(self):
            self.tick_labels_with_unit = tick_labels_with_unit
            # Recompute the Quantity subclasses at instantiation (rather than
            # once at class-definition time) so subclasses defined after this
            # module is first imported are also covered.
            self._all_issubclass_quantity = all_issubclass(Quantity)
            # Keep track of original converter in case the context manager is
            # used in a nested way.
            self._original_converter = {}

            for cls in self._all_issubclass_quantity:
                self._original_converter[cls] = munits_registry.get(cls)
                munits_registry[cls] = self

        def __enter__(self):
            return self

        def __exit__(self, type, value, tb):
            for cls in self._all_issubclass_quantity:
                if self._original_converter[cls] is None:
                    del munits_registry[cls]
                else:
                    munits_registry[cls] = self._original_converter[cls]

    return MplQuantityConverter()
