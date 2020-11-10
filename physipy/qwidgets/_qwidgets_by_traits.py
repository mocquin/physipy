from physipy import quantify

from traitlets import (
    Instance, Unicode, CFloat, Bool, CaselessStrEnum, Tuple, TraitError, validate, default, 
)
from traitlets.traitlets import _validate_bounds

from ipywidgets.widgets.widget_description import DescriptionWidget
from ipywidgets.widgets.trait_types import InstanceDict, NumberFormat
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget import register, widget_serialization
from ipywidgets.widgets.widget_core import CoreWidget
from ipywidgets.widgets.widget_int import ProgressStyle



from ast import literal_eval
import contextlib
import inspect
import os
import re
import sys
import types
import enum
from warnings import warn, warn_explicit

from traitlets.utils.getargspec import getargspec
from traitlets.utils.importstring import import_item
from traitlets.utils.sentinel import Sentinel
from traitlets.utils.bunch import Bunch
from traitlets.utils.descriptions import describe, class_of, add_article, repr_type
from traitlets import TraitType

SequenceTypes = (list, tuple, set, frozenset)

# backward compatibility, use to differ between Python 2 and 3.
ClassTypes = (type,)

# exports:

__all__ = [
    "default",
    "validate",
    "observe",
    "observe_compat",
    "link",
    "directional_link",
    "dlink",
    "Undefined",
    "All",
    "NoDefaultSpecified",
    "TraitError",
    "HasDescriptors",
    "HasTraits",
    "MetaHasDescriptors",
    "MetaHasTraits",
    "BaseDescriptor",
    "TraitType",
    "parse_notifier_name",
]

# any TraitType subclass (that doesn't start with _) will be added automatically

#-----------------------------------------------------------------------------
# Basic classes
#-----------------------------------------------------------------------------


Undefined = Sentinel('Undefined', 'traitlets',
'''
Used in Traitlets to specify that no defaults are set in kwargs
'''
)



def _validate_bounds(trait, obj, value):
    """
    Validate that a number to be applied to a trait is between bounds.
    If value is not between min_bound and max_bound, this raises a
    TraitError with an error message appropriate for this trait.
    """
    print("trait:", type(trait), trait, type(trait.min), type(trait.max))
    print("obj:", type(obj))
    print("value:", type(value), value)
    
    
    if trait.min is not None and value < trait.min:
        raise TraitError(
            "The value of the '{name}' trait of {klass} instance should "
            "not be less than {min_bound}, but a value of {value} was "
            "specified".format(
                name=trait.name, klass=class_of(obj),
                value=value, min_bound=trait.min))
    if trait.max is not None and value > trait.max:
        raise TraitError(
            "The value of the '{name}' trait of {klass} instance should "
            "not be greater than {max_bound}, but a value of {value} was "
            "specified".format(
                name=trait.name, klass=class_of(obj),
                value=value, max_bound=trait.max))
    return value

print(type(-Quantity(float('inf'), Dimension(None))))
print(type(Quantity(float('inf'), Dimension(None))))
print(type(Quantity(1000, Dimension(None))))

class QLFloat(TraitType):
    """A quantity trait."""
    
    default_value = Quantity(0.0, Dimension("L"))
    info_text = "a quantity"
    
    def __init__(self, default_value=Undefined, allow_none=False, **kwargs):
        self.min = kwargs.pop('min', Quantity(-float("inf"), Dimension("L")))
        self.max = kwargs.pop('max', Quantity(float('inf'), Dimension("L")))
        super(QLFloat, self).__init__(default_value=default_value,
                                    allow_none=allow_none, **kwargs)
        
    def validate(self, obj, value):
        if isinstance(value, int):
            value = quantify(value)
        if not isinstance(value, Quantity):
            self.error(obj, value)
        return _validate_bounds(self, obj, value)

#class Float(TraitType):
#    """A float trait."""
#
#    default_value = 0.0
#    info_text = 'a float'
#
#    def __init__(self, default_value=Undefined, allow_none=False, **kwargs):
#        self.min = kwargs.pop('min', -float('inf'))
#        self.max = kwargs.pop('max', float('inf'))
#        super(Float, self).__init__(default_value=default_value,
#                                    allow_none=allow_none, **kwargs)
#
#    def validate(self, obj, value):
#        if isinstance(value, int):
#            value = float(value)
#        if not isinstance(value, float):
#            self.error(obj, value)
#        return _validate_bounds(self, obj, value)
#
#    def from_string(self, s):
#        if self.allow_none and s == 'None':
#            return None
#        return float(s)

class QLCFloat(QLFloat):
    """A casting version of the Qfloat trait."""
    
    def validate(self, obj, value):
        try:
            value = quantify(value)
        except Exception:
            self.error(obj, value)
        return _validate_bounds(self, obj, value)

# class CFloat(Float):
#     """A casting version of the float trait."""
# 
#     def validate(self, obj, value):
#         try:
#             value = float(value)
#         except Exception:
#             self.error(obj, value)
#         return _validate_bounds(self, obj, value)

class _QLFloat(DescriptionWidget, ValueWidget, CoreWidget):
    value = QLCFloat(Quantity(0.0, Dimension("L")), help="QLFloat value").tag(sync=True)
    
    def __init__(self, value=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        super().__init__(**kwargs)

#class _Float(DescriptionWidget, ValueWidget, CoreWidget):
#    value = CFloat(0.0, help="Float value").tag(sync=True)
#
#    def __init__(self, value=None, **kwargs):
#        if value is not None:
#            kwargs['value'] = value
#        super().__init__(**kwargs)

class _BoundedQLFloat(_QLFloat):
    max = QLCFloat(Quantity(100.0, Dimension("L")), help="Max value").tag(sync=True)
    min = QLCFloat(Quantity(0.0, Dimension("L")), help="Min value").tag(sync=True)
    

    @validate('value')
    def _validate_value(self, proposal):
        """Cap and floor value"""
        value = proposal['value']
        if self.min > value or self.max < value:
            value = min(max(value, self.min), self.max)
        return value

    @validate('min')
    def _validate_min(self, proposal):
        """Enforce min <= value <= max"""
        min = proposal['value']
        if min > self.max:
            raise TraitError('Setting min > max')
        if min > self.value:
            self.value = min
        return min

    @validate('max')
    def _validate_max(self, proposal):
        """Enforce min <= value <= max"""
        max = proposal['value']
        if max < self.min:
            raise TraitError('setting max < min')
        if max < self.value:
            self.value = max
        return max
    
#class _BoundedFloat(_Float):
#    max = CFloat(100.0, help="Max value").tag(sync=True)
#    min = CFloat(0.0, help="Min value").tag(sync=True)
#
#    @validate('value')
#    def _validate_value(self, proposal):
#        """Cap and floor value"""
#        value = proposal['value']
#        if self.min > value or self.max < value:
#            value = min(max(value, self.min), self.max)
#        return value
#
#    @validate('min')
#    def _validate_min(self, proposal):
#        """Enforce min <= value <= max"""
#        min = proposal['value']
#        if min > self.max:
#            raise TraitError('Setting min > max')
#        if min > self.value:
#            self.value = min
#        return min
#
#    @validate('max')
#    def _validate_max(self, proposal):
#        """Enforce min <= value <= max"""
#        max = proposal['value']
#        if max < self.min:
#            raise TraitError('setting max < min')
#        if max < self.value:
#            self.value = max
#        return max

from ipywidgets.widgets.widget_int import IntText, BoundedIntText, IntSlider, IntProgress, IntRangeSlider, Play, SliderStyle

@register
class QLFloatSlider(_BoundedQLFloat):
    _view_name = Unicode('QLFloatSliderView').tag(sync=True)
    _model_name = Unicode('QLFloatSliderModel').tag(sync=True)
    step = QLCFloat(Quantity(0.1, Dimension("L")), allow_none=True, help="Minimum step to increment the value").tag(sync=True)
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'],
        default_value='horizontal', help="Vertical or horizontal.").tag(sync=True)
    readout = Bool(True, help="Display the current value of the slider next to it.").tag(sync=True)
    readout_format = NumberFormat(
        '.2f', help="Format for the readout").tag(sync=True)
    continuous_update = Bool(True, help="Update the value of the widget as the user is holding the slider.").tag(sync=True)
    disabled = Bool(False, help="Enable or disable user changes").tag(sync=True)

    style = InstanceDict(SliderStyle).tag(sync=True, **widget_serialization)

    
#@register
#class FloatSlider(_BoundedFloat):
#    """ Slider/trackbar of floating values with the specified range.
#    Parameters
#    ----------
#    value : float
#        position of the slider
#    min : float
#        minimal position of the slider
#    max : float
#        maximal position of the slider
#    step : float
#        step of the trackbar
#    description : str
#        name of the slider
#    orientation : {'horizontal', 'vertical'}
#        default is 'horizontal', orientation of the slider
#    readout : {True, False}
#        default is True, display the current value of the slider next to it
#    readout_format : str
#        default is '.2f', specifier for the format function used to represent
#        slider value for human consumption, modeled after Python 3's format
#        specification mini-language (PEP 3101).
#    """
#    _view_name = Unicode('FloatSliderView').tag(sync=True)
#    _model_name = Unicode('FloatSliderModel').tag(sync=True)
#    step = CFloat(0.1, allow_none=True, help="Minimum step to increment the value").tag(sync=True)
#    orientation = CaselessStrEnum(values=['horizontal', 'vertical'],
#        default_value='horizontal', help="Vertical or horizontal.").tag(sync=True)
#    readout = Bool(True, help="Display the current value of the slider next to it.").tag(sync=True)
#    readout_format = NumberFormat(
#        '.2f', help="Format for the readout").tag(sync=True)
#    continuous_update = Bool(True, help="Update the value of the widget as the user is holding the slider.").tag(sync=True)
#    disabled = Bool(False, help="Enable or disable user changes").tag(sync=True)
#
#    style = InstanceDict(SliderStyle).tag(sync=True, **widget_serialization)

QLFloatSlider(2*m, min=0*m, max=10*m)