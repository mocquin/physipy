import sys
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
from physipy import m, s, Quantity, Dimension, rad, units
from numpy import pi

import ipywidgets as ipyw
import traitlets
from physipy.qwidgets.qwidgets import QuantityFloatSlider

mm = units["mm"]

class QuantitySlider(ipyw.Box):
    qvalue = traitlets.Instance(Quantity, allow_none=True)
    number = traitlets.Float(allow_none=True)
    display_val = traitlets.Unicode(allow_none=True)

    def __init__(self, qvalue, **kwargs):
        self.slider = ipyw.FloatSlider(readout=False)
        self.label = ipyw.Label()
        super().__init__(**kwargs)
        self.children = [self.slider, self.label]
        traitlets.link((self.slider, 'value'), (self, 'number'))
        traitlets.link((self.label, 'value'),  (self, 'display_val'))
        # set qvalue
        self.qvalue = qvalue
        # Maybe add a callback here that is triggered on changes in self.slider.value?
        def update_qvalue_from_slider_change(change):
            self.qvalue = Quantity(change.new, self.qvalue.dimension)
            # other way : but the display_val update in the last line becomes useless because only
            # qvalue.value changes, not qvalue
            #self.qvalue.value = change.new
            #self.display_val = f'{self.qvalue}'
        self.slider.observe(update_qvalue_from_slider_change, names="value")

    @traitlets.observe('qvalue')
    def _update_derived_traitlets(self, proposal):
        self.number = self.qvalue.value
        self.display_val = f'{self.qvalue}'


qs = QuantitySlider(3*m)
qs


# +
class QuantitySlider(ipyw.Box):
    qvalue = traitlets.Instance(Quantity, allow_none=True)

    def __init__(self, qvalue, **kwargs):
        self.slider = ipyw.FloatSlider(readout=False)
        self.label = ipyw.Label()
        super().__init__(**kwargs)
        self.children = [self.slider, self.label]
        self.qvalue = qvalue
        def update_qvalue_from_slider_change(change):
            self.qvalue = Quantity(change.new, self.qvalue.dimension)
        self.slider.observe(update_qvalue_from_slider_change, names="value")

    @traitlets.observe('qvalue')
    def _update_derived_traitlets(self, proposal):
        self.slider.value = self.qvalue.value
        self.label.value = f'{self.qvalue}'

qs = QuantitySlider(3*m)
qs
# -


