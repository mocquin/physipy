# +

from physipy import m, s, Quantity, Dimension, rad, units, K
from numpy import pi

# +
import ipywidgets as ipyw
import traitlets

print(ipyw.__version__)
# -


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

qw





# +
from ipywidgets import Layout
from traitlets import TraitError
from physipy import quantify, Dimension

class QuantitySliderWithBounds(ipyw.Box, ipyw.ValueWidget, ipyw.DOMWidget):
    # dimension trait : a Dimension instance
    dimension = traitlets.Instance(Dimension, allow_none=False)
    # value trait : a Quantity instance
    value = traitlets.Instance(Quantity, allow_none=False)
    # qmin, qmax, qstep : params of slider, Quantity instances
    qmin = traitlets.Instance(Quantity, allow_none=False)
    qmax = traitlets.Instance(Quantity, allow_none=False)
    qstep = traitlets.Instance(Quantity, allow_none=False)
    # value_number, min_number, max_number, step_number : slider's values
    value_number = traitlets.Float(allow_none=True)
    min_number = traitlets.Float(allow_none=True)
    max_number = traitlets.Float(allow_none=True)
    step_number = traitlets.Float(allow_none=True)
    # string trait for label
    display_val = traitlets.Unicode(allow_none=True)
    description = traitlets.Unicode(allow_none=True)

    def __init__(self, value=0.0, qmin=None, qmax=None, qstep=None, disabled=False, 
                 orientation="horizontal", readout_format=".1f", 
                 continuous_update=True, **kwargs):
        
        
        #base_qs = {"value":value, "qmin":min, "qmax":max, "qstep":step}
        #base_qs_none = {k:v for base_qs.items() if v is None}
        #
        #
        ######### CHECKING DIMENSION
        #to_check_dim = [v for k,v in base_qs.items() if v is not None]
        ## all are none
        #if len(to_check_dim) == 0:
        #    self.value = quantity(value)
        #    self.qmin = Quantity(0.0, self.value.dimension)
        #    self.qmax = Quantity(100.0, self.value.dimension)
        #    self.qstep = Quantity(0.1, self.value.dimension)            
        ## all are none except one
        #elif len(to_check_dim) == 1:
        #    pass
        #else:
        #    qs = [quantify(x) for x in to_check_dim]
        #    base_dim = qs[0].dimension
        #    for x in to_check_dim[1:]:
        #        if not x.dimension == base_dim:
        #            raise DimensionError(x.dimension, base_dim)
        #            
        #
        ######## INIT QUANTITIES
        ## at this point, all inputs that are not None, in to_check_dim, were dimenion-checked
        ## so its safe to set them, using a quantify
        #for k, v in to_check_dim.items():
        #    v = quantify(v)
        #    self.__setattr__(k, v)
        ## setting quantities that were None
        #for k, v in base_qs_none:
        #    if k == "value":
                
        
            
        # set qvalue
        #if value is None:
        #    self.value = Quantity(0, Dimension(None))
        #else:
        
        
        value = quantify(value)
        self.dimension = value.dimension

        #    self.value = value
        self.value = value
        #self.qmin = quantify(min)
        #self.qmax = quantify(max)
        #self.qstep = quantify(step)

        # min value
        if qmin is not None:
            self.qmin = quantify(qmin)
            if not qmin.dimension == self.value.dimension:
                raise DimensionError(qmin.dimension, self.value.dimension)
        else:
            self.qmin = Quantity(0.0, self.value.dimension)
        # max value
        if qmax is not None:
            self.qmax = quantify(qmax)
            if not qmax.dimension == self.value.dimension:
                raise DimensionError(qmax.dimension, self.value.dimension)
        else:
            self.qmax = Quantity(100.0, self.value.dimension)        
        # step value
        if qstep is not None:
            self.qstep = quantify(qstep)
            if not self.qstep.dimension == self.value.dimension:
                raise DimensionError(self.qstep.dimension, self.value.dimension)
        else:
            self.qstep = Quantity(0.1, self.value.dimension)     
            
        #qmin.favunit = self.value.favunit
        #qmax.favunit = self.value.favunit
        
        
        #self.slider = ipyw.FloatSlider(readout=False)
        #self.label = ipyw.Label()
        
        
        self.slider = ipyw.FloatSlider(value=self.value.value,
                                            min=self.qmin.value,
                                            max=self.qmax.value,
                                            step=self.qstep.value,
                                            description=self.description,
                                            disabled=disabled,
                                            continuous_update=continuous_update,
                                            orientation=orientation,
                                            readout=False,  # to disable displaying readout value
                                            readout_format=readout_format,
                                            layout=Layout(width="30%",
                                                          margin="0px",
                                                          border="solid #3295a8"))

        self.label = ipyw.Label(value=str(self.value))
        
        
        
        super().__init__(**kwargs)
        self.children = [self.slider, self.label]
        
        # link between quantities and slider values
        traitlets.link((self.slider, 'value'), (self,'value_number'))
        traitlets.link((self.slider, "min")  , (self,  "min_number"))
        traitlets.link((self.slider, "max")  , (self,  "max_number"))
        traitlets.link((self.slider, "step") , (self, "step_number"))
        # link between quantity and label
        traitlets.link((self.label, 'value'),  (self, 'display_val'))

        # Maybe add a callback here that is triggered on changes in self.slider.value?
        def update_value_from_slider_change(change):
            self.value = Quantity(change.new, self.value.dimension)
            # other way : but the display_val update in the last line becomes useless because only
            # qvalue.value changes, not qvalue
            #self.qvalue.value = change.new
            #self.display_val = f'{self.qvalue}'
        self.slider.observe(update_value_from_slider_change, names="value")

        
    # update display_val and value_number on quantity change
    @traitlets.observe('value')
    def _update_derived_traitlets(self, proposal):
        self.value_number = self.value.value
        self.display_val = f'{str(self.value)}'
        
    # update min_number on quantity change
    @traitlets.observe('qmin')
    def _update_min_slider(self, proposal):
        self.min_number = self.qmin.value
        
    @traitlets.observe('qmax')
    def _update_max_slider(self, proposal):
        self.max_number = self.qmax.value
        
    @traitlets.observe('qstep')
    def _update_step_slider(self, proposal):
        self.step_number = self.qstep.value
        
    @traitlets.validate('value')
    def _valid_value(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise TraitError('value and parity should be consistent')
        return proposal['value']
    @traitlets.validate('qmax')
    def _valid_qmax(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise DimensionError(proposal['value'].dimension, self.dimension)
        return proposal['value']
    @traitlets.validate('qmin')
    def _valid_qmax(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise DimensionError(proposal['value'].dimension, self.dimension)
        return proposal['value']
    @traitlets.validate('qstep')
    def _valid_qstep(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise DimensionError(proposal['value'].dimension, self.dimension)
        return proposal['value']
    
# -

# interact without abbreviation
qs = QuantitySliderWithBounds(value=4*m, qmin=10*m, qmax=20*m, qstep=2*m)
qs

# +


def toto(x):
    return str(x*2)

ipyw.interact(toto, x=qs);
# -

# ## boxing

qs = QuantitySliderWithBounds(value=4*m, qmin=10*m, qmax=20*m, qstep=2*m)
qs

ipyw.VBox([qs, qs])

# ## interactive

# interact without abbreviation
qs = QuantitySliderWithBounds(value=4*m, qmin=10*m, qmax=20*m, qstep=2*m)
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
    print(2*i)

    
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
wa = QuantitySliderWithBounds(value=4*m, qmin=10*m, qmax=20*m, qstep=2*m)
wb = QuantitySliderWithBounds(value=1*s, qmin=10*s, qmax=20*s, qstep=2*s)
wc = QuantitySliderWithBounds(value=4*K, qmin=10*K, qmax=20*K, qstep=2*K)

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook 
    # output area.
    print((a, b, c))
    print(a*b/c)

out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
# -

wb.qmin = 0.01*s

# ## link

# +
qw1 = QuantitySliderWithBounds(3*m, qmin=10*m, qmax=20*m, qstep=2*m)
qw2 = QuantitySliderWithBounds(5*m, qmin=10*m, qmax=20*m, qstep=2*m)


mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
# -

qw1

qw2

# ## observe

# +
    
qw = QuantitySliderWithBounds(2*m,qmin=10*m, qmax=20*m, qstep=2*m)


square_display = ipyw.HTML(description="Square: ",
                              value='{}'.format(qw.value**2))

def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)

qw.observe(update_square_display, names='value')

ipyw.VBox([qw, square_display])

# -








# +
class QuantityText(ipyw.Box, ipyw.ValueWidget, ipyw.DOMWidget):
    # dimension trait : a Dimension instance
    dimension = traitlets.Instance(Dimension, allow_none=False)
    # value trait : a Quantity instance
    value = traitlets.Instance(Quantity, allow_none=False)
    # value_number : float value of quantity
    value_number = traitlets.Float(allow_none=True)
    # string trait text
    display_val = traitlets.Unicode(allow_none=True)
    # description
    description = traitlets.Unicode(allow_none=True)
    
    def __init__(self, value=0.0, disabled=False, 
                 continuous_update=True, description="Quantity:",
                 fixed_dimension=False,
                 **kwargs):
        
        # context for parsing
        self.context = {**units, "pi":pi}
        self.description = description
        

        # quantity work
        # set dimension
        value = quantify(value)
        self.dimension = value.dimension
        # if true, any change in value must have same dimension as initial dimension
        self.fixed_dimension = fixed_dimension
        
        # set quantity
        self.value = value
        # set text widget
        self.text = ipyw.Text(value=str(self.value),
                              placeholder='Type python exp',
                              description=self.description,#'Set to:',
                              disabled=disabled,
                              continuous_update=continuous_update,
                              layout=Layout(width='25%',
                                            margin="0px 0px 0px 0px",
                                            padding="0px 0px 0px 0px",
                                            border="solid gray"))
        
        # link text value and display_val unicode trait
        traitlets.link((self.text, "value"), (self, "display_val"))
        super().__init__(**kwargs)
        self.children = [self.text]

        # on_submit observe
        def text_update_values(wdgt):
            # get expression entered
            expression = wdgt.value
            # eval expression with unit context
            try:
                res = eval(expression, self.context)
                res = quantify(res)
                # update quantity value
                self.value = res
                # update display_value
                self.display_val = str(self.value)
            except:
                self.display_val = str(self.value)
        self.text.on_submit(text_update_values)

    # update value_number and text on quantity value change
    @traitlets.observe("value")
    def _update_display_val(self, proposal):
        self.value_number = self.value.value
        self.dimension = self.value.dimension
        self.display_val = f'{str(self.value)}'

    @traitlets.validate('value')
    def _valid_value(self, proposal):
        if self.fixed_dimension and proposal['value'].dimension != self.dimension:
            raise TraitError('value and parity should be consistent')
        return proposal['value']
    

class FDQuantityText(QuantityText):
    
    def __init__(self, value=0.0, disabled=False, 
                 continuous_update=True, description="Quantity:", *args,
                 **kwargs):
        
        super().__init__(value=value, disabled=disabled,
                         continuous_update=continuous_update, 
                         description=description,
                         fixed_dimension=True, 
                         *args, 
                         **kwargs)



# -

w = QuantityText()
w

# w.value returns the quantity
print(type(w.value))
print(w.value)
print(w.description)
# value_number is a Float trait that is linked to the quantity value
print(w.value_number)
print(w.value.value)
# dimension is linked to the quantity dimension
print(w.dimension)
print(w.value.dimension)

w = QuantityText(fixed_dimension=True)
w

print(w.value)

w = FDQuantityText()
w

# w.value returns the quantity
print(type(w.value))
print(w.value)
print(w.description)
# value_number is a Float trait that is linked to the quantity value
print(w.value_number)
print(w.value.value)
# dimension is linked to the quantity dimension
print(w.dimension)
print(w.value.dimension)

# ## interact without abbreviation

qs =  QuantityText(2*m)
qs

# +


def toto(x):
    return str(x*2)

ipyw.interact(toto, x=qs);
# -

# ## boxing

qs = QuantityText(1*m*K)
qs

ipyw.VBox([qs, qs])

# ## interactive

# interact without abbreviation
qs =  QuantityText()
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
    print(2*i)

    
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
wa = QuantityText()
wb = QuantityText()
wc = QuantityText()

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook 
    # output area.
    print((a, b, c))
    print(a*b/c)

out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
# -

wb.value = 2*m

# ## link

# +
qw1 =  QuantityText()
qw2 =  QuantityText()


mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
# -

qw1

qw2

# ## observe

# +
    
qw =  QuantityText()


square_display = ipyw.HTML(description="Square: ",
                              value='{}'.format(qw.value**2))

def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)

qw.observe(update_square_display, names='value')

ipyw.VBox([qw, square_display])

# -




ipyw.Textarea(
    value='Hello World',
    placeholder='Type something',
    description='String:',
    disabled=False
)
ipyw.Text(
    value='Hello World',
    placeholder='Type something',
    description='String:',
    disabled=False
)

x = ipyw.FloatText(
    value=7.5,
    description='Any:',
    disabled=False
)
x

ipyw.BoundedFloatText(
    value=7.5,
    min=0,
    max=10.0,
    step=0.1,
    description='Text:',
    disabled=False
)


class QuantityWidget(ipyw.VBox, ipyw.ValueWidget, ipyw.DOMWidget):
    value = traitlets.Instance(Quantity)
    description = traitlets.Unicode()
    
    def __init__(self, value=0.0, qmin=None, qmax=None, qstep=None,
                 disabled=False, continuous_update=True, description="QuantityText description", 
                 **kwargs):
        
        self.qtext = QuantityText(value)
        self.qslider = QuantitySliderWithBounds(value, qmin, qmax, qstep)
        self.value = self.qtext.value
        self.description = description
        traitlets.link((self.qtext, ("value")), 
                       (self.qslider, ("value")))
        traitlets.link((self, 'value'),
                       (self.qtext, "value"))
        
        super().__init__(**kwargs)
        self.children = [self.qtext, self.qslider]
        


qw = QuantityWidget(2*m)
qw

print(qw.value)



# ## interact without abbreviation

qs =  QuantityWidget(2*m)
qs

# +


def toto(x):
    return str(x*2)

ipyw.interact(toto, x=qs);
# -

# ## boxing

qs =  QuantityWidget(2*m)
qs

ipyw.VBox([qs, qs])

# ## interactive

# interact without abbreviation
qs = QuantityWidget(2*m)
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
    print(2*i)

    
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
wa =  QuantityWidget(2*m)
wb =  QuantityWidget(2*m)
wc =  QuantityWidget(2*m)

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook 
    # output area.
    print((a, b, c))
    print(a*b/c)

out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
# -

wb.value = 2*m

# ## link

# +
qw1 =  QuantityWidget(2*m)
qw2 = QuantityWidget(2*m)


mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
# -

qw1

qw2

# ## observe

# +
    
qw =  QuantityWidget(2*m)


square_display = ipyw.HTML(description="Square: ",
                              value='{}'.format(qw.value**2))

def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)

qw.observe(update_square_display, names='value')

ipyw.VBox([qw, square_display])

# -
# ## interact without abbreviation









# +

#       self.favunit_str_list = [u_str for u_str, u_q in units.items() ]#if self.dimension == u_q.dimension]
#       self.favunit_str_list.append("-")
#       self.favunit_dd = ipyw.Dropdown(
#                           options=self.favunit_str_list,
#                           value=str(self.favunit.symbol),
#                           description='Favunit:',
#                           layout=Layout(width="20%",
#                                         padding="0px 0px 0px 0px",
#                                         margin="0px 0px 0px 0px",
#                                         border="solid red")
#                                   )


# +
from physipy import dimensionify

class FavunitDropdown(ipyw.Box, ipyw.DOMWidget):
    dimension = traitlets.Instance(Dimension)
    qfavunit = traitlets.Instance(Quantity, allow_none=True)
    strfavunit = traitlets.Unicode()
    
    def __init__(self, dimension=None, all_units=False,
                 **kwargs):
        
        # pouvoir ajouter des favunits lors de l'initiatsation dans la liste

        self.dimension = dimensionify(dimension)
        self.units = units
        self.units["-"] = Quantity(1, Dimension(None), symbol="-")
        
        # list of available units
        if self.dimension == Dimension(None) or all_units:
            self.favunit_str_list = [u_str for u_str, u_q in self.units.items() ]
       # elif not all_units and self.dimension is not None:
       #     self.favunit_str_list = [u_str for u_str, u_q in self.units.items() if self.dimension == u_q.dimension]
        else:
            self.favunit_str_list = [u_str for u_str, u_q in self.units.items() if self.dimension == u_q.dimension ]
        self.favunit_str_list.append("-")
        self.qfavunit = self.units["-"]
        
        # dropdown
        self.favunit_dd = ipyw.Dropdown(
                           options=self.favunit_str_list,
                           value=str(self.qfavunit.symbol),
                           description='Favunit:',
                           layout=Layout(width="30%",
                                         margin="5px 5px 5px 5px",                                         
                                         border="solid black")
                                   )
        
        super().__init__(**kwargs)
        self.children = [self.favunit_dd]
        
        
        ### 3. Change favunit
        # selection of favunit
        def update_favunit_on_favunit_dd_change(change):
            # retrieve new favunit q
            self.qfavunit = self.units[change.new]
        self.favunit_dd.observe(update_favunit_on_favunit_dd_change, names="value")
# -

w = FavunitDropdown()
w

print(w.qfavunit)
print(type(w.qfavunit))
print(type(w.qfavunit.symbol))
print(w.qfavunit.symbol)

w = FavunitDropdown(m)
w

print(w.qfavunit)
print(type(w.qfavunit))
print(type(w.qfavunit.symbol))
print(w.qfavunit.symbol)

w = FavunitDropdown(Dimension("L"))
w

print(w.qfavunit)
print(type(w.qfavunit))
print(type(w.qfavunit.symbol))
print(w.qfavunit.symbol)







# +
from ipywidgets import Layout
from traitlets import TraitError
from physipy import quantify, Dimension

class QuantitySliderWithTextBounds(ipyw.Box, ipyw.ValueWidget, ipyw.DOMWidget):
    # dimension trait : a Dimension instance
    dimension = traitlets.Instance(Dimension, allow_none=False)
    # value trait : a Quantity instance
    value = traitlets.Instance(Quantity, allow_none=False)
    # qmin, qmax, qstep : params of slider, Quantity instances
    qmin = traitlets.Instance(Quantity, allow_none=False)
    qmax = traitlets.Instance(Quantity, allow_none=False)
    qstep = traitlets.Instance(Quantity, allow_none=False)
    # value_number, min_number, max_number, step_number : slider's values
    value_number = traitlets.Float(allow_none=True)
    min_number = traitlets.Float(allow_none=True)
    max_number = traitlets.Float(allow_none=True)
    step_number = traitlets.Float(allow_none=True)
    # string trait for label
    display_val = traitlets.Unicode(allow_none=True)
    display_min = traitlets.Unicode(allow_none=True)
    display_max = traitlets.Unicode(allow_none=True)
    description = traitlets.Unicode(allow_none=True)

    def __init__(self, value=0.0, qmin=None, qmax=None, qstep=None, disabled=False, 
                 orientation="horizontal", readout_format=".1f", 
                 continuous_update=True, **kwargs):
        
        
        #base_qs = {"value":value, "qmin":min, "qmax":max, "qstep":step}
        #base_qs_none = {k:v for base_qs.items() if v is None}
        #
        #
        ######### CHECKING DIMENSION
        #to_check_dim = [v for k,v in base_qs.items() if v is not None]
        ## all are none
        #if len(to_check_dim) == 0:
        #    self.value = quantity(value)
        #    self.qmin = Quantity(0.0, self.value.dimension)
        #    self.qmax = Quantity(100.0, self.value.dimension)
        #    self.qstep = Quantity(0.1, self.value.dimension)            
        ## all are none except one
        #elif len(to_check_dim) == 1:
        #    pass
        #else:
        #    qs = [quantify(x) for x in to_check_dim]
        #    base_dim = qs[0].dimension
        #    for x in to_check_dim[1:]:
        #        if not x.dimension == base_dim:
        #            raise DimensionError(x.dimension, base_dim)
        #            
        #
        ######## INIT QUANTITIES
        ## at this point, all inputs that are not None, in to_check_dim, were dimenion-checked
        ## so its safe to set them, using a quantify
        #for k, v in to_check_dim.items():
        #    v = quantify(v)
        #    self.__setattr__(k, v)
        ## setting quantities that were None
        #for k, v in base_qs_none:
        #    if k == "value":
                
        
            
        # set qvalue
        #if value is None:
        #    self.value = Quantity(0, Dimension(None))
        #else:
        
        
        value = quantify(value)
        self.dimension = value.dimension

        #    self.value = value
        self.value = value
        #self.qmin = quantify(min)
        #self.qmax = quantify(max)
        #self.qstep = quantify(step)

        # min value
        if qmin is not None:
            self.qmin = quantify(qmin)
            if not qmin.dimension == self.value.dimension:
                raise DimensionError(qmin.dimension, self.value.dimension)
        else:
            self.qmin = Quantity(0.0, self.value.dimension)
        # max value
        if qmax is not None:
            self.qmax = quantify(qmax)
            if not qmax.dimension == self.value.dimension:
                raise DimensionError(qmax.dimension, self.value.dimension)
        else:
            self.qmax = Quantity(100.0, self.value.dimension)        
        # step value
        if qstep is not None:
            self.qstep = quantify(qstep)
            if not self.qstep.dimension == self.value.dimension:
                raise DimensionError(self.qstep.dimension, self.value.dimension)
        else:
            self.qstep = Quantity(0.1, self.value.dimension)     
            
        #qmin.favunit = self.value.favunit
        #qmax.favunit = self.value.favunit
        
        
        #self.slider = ipyw.FloatSlider(readout=False)
        #self.label = ipyw.Label()
        
        
        self.slider = ipyw.FloatSlider(value=self.value.value,
                                            min=self.qmin.value,
                                            max=self.qmax.value,
                                            step=self.qstep.value,
                                            description=self.description,
                                            disabled=disabled,
                                            continuous_update=continuous_update,
                                            orientation=orientation,
                                            readout=False,  # to disable displaying readout value
                                            readout_format=readout_format,
                                            layout=Layout(width="30%",
                                                          margin="0px",
                                                          border="solid #3295a8"))

        self.label = ipyw.Label(value=str(self.value))
        
        
        self.qtextmin = QuantityText(qmin)
        self.qtextmax = QuantityText(qmax)

        self.qslider = QuantitySliderWithBounds(value, qmin, qmax, qstep)
        self.value = self.qtext.value
        self.description = description
        traitlets.link((self.qtext, ("value")), 
                       (self.qslider, ("value")))
        traitlets.link((self, 'value'),
                       (self.qtext, "value"))
        

        
    
        super().__init__(**kwargs)
        self.children = [self.qtextmin,
                         self.slider, 
                         self.qtextmax,
                         self.label]
        
        # link between quantities and slider values
        traitlets.link((self.slider, 'value'), (self,'value_number'))
        traitlets.link((self.slider, "min")  , (self,  "min_number"))
        traitlets.link((self.slider, "max")  , (self,  "max_number"))
        traitlets.link((self.slider, "step") , (self, "step_number"))
        # link between quantity and label
        traitlets.link((self.label, 'value'),  (self, 'display_val'))

        # Maybe add a callback here that is triggered on changes in self.slider.value?
        def update_value_from_slider_change(change):
            self.value = Quantity(change.new, self.value.dimension)
            # other way : but the display_val update in the last line becomes useless because only
            # qvalue.value changes, not qvalue
            #self.qvalue.value = change.new
            #self.display_val = f'{self.qvalue}'
        self.slider.observe(update_value_from_slider_change, names="value")

        
    # update display_val and value_number on quantity change
    @traitlets.observe('value')
    def _update_derived_traitlets(self, proposal):
        self.value_number = self.value.value
        self.display_val = f'{str(self.value)}'
        
    # update min_number on quantity change
    @traitlets.observe('qmin')
    def _update_min_slider(self, proposal):
        self.min_number = self.qmin.value
        
    @traitlets.observe('qmax')
    def _update_max_slider(self, proposal):
        self.max_number = self.qmax.value
        
    @traitlets.observe('qstep')
    def _update_step_slider(self, proposal):
        self.step_number = self.qstep.value
        
    @traitlets.validate('value')
    def _valid_value(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise TraitError('value and parity should be consistent')
        return proposal['value']
    @traitlets.validate('qmax')
    def _valid_qmax(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise DimensionError(proposal['value'].dimension, self.dimension)
        return proposal['value']
    @traitlets.validate('qmin')
    def _valid_qmax(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise DimensionError(proposal['value'].dimension, self.dimension)
        return proposal['value']
    @traitlets.validate('qstep')
    def _valid_qstep(self, proposal):
        if proposal['value'].dimension != self.dimension:
            raise DimensionError(proposal['value'].dimension, self.dimension)
        return proposal['value']
    
# -



