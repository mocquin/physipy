# +

from physipy import m, s, Quantity, Dimension, rad, units, K
from numpy import pi
# -

import ipywidgets as ipyw
import traitlets


mm = units["mm"]



class QuantitySlider(ipyw.Box, ipyw.ValueWidget, ipyw.DOMWidget):
    value = traitlets.Instance(Quantity, allow_none=False)
    number = traitlets.Float(allow_none=True)
    display_val = traitlets.Unicode(allow_none=True)
    description = traitlets.Unicode(allow_none=True)

    def __init__(self, value, **kwargs):
        self.slider = ipyw.FloatSlider(readout=False)
        self.label = ipyw.Label()
        super().__init__(**kwargs)
        self.children = [self.slider, self.label]
        traitlets.link((self.slider, 'value'), (self, 'number'))
        traitlets.link((self.label, 'value'),  (self, 'display_val'))
        # set qvalue
        self.value = value
        # Maybe add a callback here that is triggered on changes in self.slider.value?
        def update_value_from_slider_change(change):
            self.value = Quantity(change.new, self.value.dimension)
            # other way : but the display_val update in the last line becomes useless because only
            # qvalue.value changes, not qvalue
            #self.qvalue.value = change.new
            #self.display_val = f'{self.qvalue}'
        self.slider.observe(update_value_from_slider_change, names="value")

    @traitlets.observe('value')
    def _update_derived_traitlets(self, proposal):
        self.number = self.value.value
        self.display_val = f'{str(self.value)}'



# ## interact witjout abbreviation
#
# https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
#
# Finally, if you need more granular control than that afforded by the abbreviation, you can pass a ValueWidget instance as the argument. A ValueWidget is a widget that aims to control a single value. Most of the widgets bundled with ipywidgets inherit from ValueWidget. For more information, see this section on widget types.
#
# Deadlink in the doc : https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Custom.ipynb#DOMWidget,-ValueWidget-and-Widget
#
# ValueWidget code : https://github.com/jupyter-widgets/ipywidgets/blob/master/ipywidgets/widgets/valuewidget.py
#
#
# ValueWidget : 
#  - inherits from Widget
#  - defines a value traitlets.Any attribute
#  
# Widget code : https://github.com/jupyter-widgets/ipywidgets/blob/51322341d4f6d88449f9dbf6d538c60de6d23543/ipywidgets/widgets/widget.py#L260
# Widget inherits from traitlets.HasTraits
#
# The widget must also have a description attribute
#
#
# At init of interactive, which inherits from VBox, https://github.com/jupyter-widgets/ipywidgets/blob/9f6d5de1025fb02e7cad16c0b0cd462614482c36/ipywidgets/widgets/interaction.py#L187
# a loop is done on widgets and check if are DOMWidget. If not, they are not added to the output hence not rendered on the front end ; can be seen in print(w.children) and comparaing with a classic slider
#
# DOMWidget also inherits from Widget
#
# To work, widget must inherit from ValueWidget
#
# So t make it work, I should create real low level widgets inheriting from DOMWIdget https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Low%20Level.html?highlight=DOMWidget#Widget-skeleton
#
# Carefull with abbreviation and iter : iter makes believe Quantity is a tuple or some kind

# interact without abbreviation
qs = QuantitySlider(3*m)
qs

# +


def toto(x):
    return str(x*2)

ipyw.interact(toto, x=qs)
# -

# ## boxing

qs = QuantitySlider(3*m)
qs

ipyw.VBox([qs, qs])

# ## interactive

# interact without abbreviation
qs = QuantitySlider(3*m)
qs


# +
def slow_function(i):
    """
    Sleep for 1 second then print the argument
    """
    from time import sleep
    print('Sleeping...')
    sleep(1)
    print(i)

w = ipyw.interactive(slow_function, i=qs)
# -

w


# +
def slow_function(i):
    """
    Sleep for 1 second then print the argument
    """
    from time import sleep
    print('Sleeping...')
    sleep(1)
    print(i)

ipyw.interact_manual(slow_function,i=qs)
# -

# ## interactive output

# +
wa = QuantitySlider(3*m)
wb = QuantitySlider(2*s)
wc = QuantitySlider(4*K)

# An HBox lays out its children horizontally
ui = ipyw.HBox([wa, wb, wc])

def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook 
    # output area.
    print((a, b, c))
    print(a*b/c)

out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
# -

# ## link

# +
qw1 = QuantitySlider(3*m)
qw2 = QuantitySlider(5*m)


mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
# -

qw1

qw2

# ## observe

# +
    
qw = QuantitySlider(2*m)


square_display = ipyw.HTML(description="Square: ",
                              value='{}'.format(qw.value**2))

def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)

qw.observe(update_square_display, names='value')

ipyw.VBox([qw, square_display])

# -









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


