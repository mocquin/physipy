import matplotlib
import matplotlib.units as munits

from .quantity import Quantity, Dimension, quantify, make_quantity, DimensionError

class QuantityConverter(munits.ConversionInterface):

    @staticmethod
    def default_units(q, axis):
        if axis.units is not None:
            if not q.dimension == axis.units.dimension:
                raise DimensionError(q.dimension, axis.units.dimension)
        q_unit = q._plot_extract_q_for_axe()
        return q_unit

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