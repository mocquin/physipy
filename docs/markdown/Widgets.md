---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Quickstart of available widgets 

```python
import ipywidgets as ipyw
from numpy import pi

from physipy import m, s, Quantity, Dimension, rad, units
from physipy.qwidgets.qipywidgets import (
    QuantityText, 
    QuantitySlider, 
    QuantityTextSlider,
    QuantityRangeSlider,
    FavunitDropdown)

mm = units["mm"]
a = 4*m
a.favunit = mm
```

```python
# Text area
qt = QuantityText(a, description="Text")
# Slider
qs = QuantitySlider(a**2, description="Slider")
# Linked Text-Slider
qts = QuantityTextSlider(a**0.5, description="Text-Slider")
# Range slider
qrs = QuantityRangeSlider(a, 10*a, label=True, description="Range-Slider")
# Dropdown
favunit_dd = FavunitDropdown()

ipyw.VBox([
    qt, 
    qs,
    qts,
    qrs,
    favunit_dd,
])
```


For most widgets, a "Fixed-Dimension" version is available, prefixed "FD". Once defined, you can't change the value to a quantity with another dimension.


Functionnalitities : 
 - VBoxing, HBoxing : `ipyw.VBox([qw, qw])`
 - interact with abbreviation : `interact(3*m)`
 - interact with widget : `interact(QuantityText(3*m))`
 - interactive : `w = ipyw.interactive(slow_function, i=qs)`
 - interactive_output : `out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})`
 - observe : 
 - link : `mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))`
 - jslink : `mylink = ipyw.jslink((qw1, 'value'), (qw2, 'value'))`


# Text


2 types that inherit from QuantityText : 
 - free QuantityText
 - Fixed-dimension "FDQuantityText"


## QuantityText
Basically a text area that will parse python expression into a numerical physical Quantity. It can be dimensionless (like 5 or 2\*pi), and can be of any dimension at any time:

```python
w = QuantityText()
w
```

A QuantityText has typical attributes of a widget and a Quantity, except the `value` is the actual Quantity, not the value of the Quantity : 

```python
print(w.value)
print(w.value.value)
#print(w.description)
print(w.fixed_dimension)
```

```python
print(w.value)
#print(w.description)
```

```python
print(w.value)
print(w.dimension)
#print(w.description)
```

With custom description : 

```python
QuantityText(description="Weight")
```

With init value

```python
w = QuantityText(2*pi*s)
w
```

```python
QuantityText(2*pi*rad, description="Angle")
```

A `fixed_dimension` attribute can be set to allow change of dimension. By default is false, and so dimension can be changed

```python
# start with seconds ...
a = QuantityText(2*pi*s)
print(a.fixed_dimension)
a
```

```python
# ... then change into radians
a.value = 2*rad
print(a.value)
print(a.fixed_dimension)
```

```python
# if fixed_dimension=True ...
b = QuantityText(2*pi*s, fixed_dimension=True)
b
```

```python
# ... cannot be changed
try:
    b.value = 2*m
except:
    print("b.fixed_dimension =", b.fixed_dimension,
          ", hence Quantity must be same dimension.")
```

```python
# handle favunit
a = 3*m
a.favunit = mm
w = QuantityTextSlider(a)
w
```

```python
w = QuantityTextSlider(3*m)
w
```

```python
w.qslider.favunit = mm
w
```

# Fixed-Dimension QuantityText
A QuantityText that will set a dimension at creation, and not allow any other dimension:


A fixed-dimensionless quantity : (trying to set a quantity with another dimension will be ignored : example : type in '5\*m' then Enter)

```python
from numpy import pi
from physipy import m
from physipy.qwidgets.qipywidgets import FDQuantityText
# init at value 0, then change its value and print
w2 = FDQuantityText()
w2
```

```python
print(w2.value)
```

A fixed-length quantity :

```python
# create with a length, then only another can be set
w3 = FDQuantityText(pi*m)
w3
# try setting a quantity with another dimension -->
```

```python
print(w3.value)
```

# Testing with QuantityText


## interact without abbreviation


The `QuantityText` widget can be passed to `ipywidgets.interact` decorator : 

```python
# define widget
qs = QuantityText(2*m)

# define function


def toto(x):
    return str(x*2)


# wrap function with interact and pass widget
res = ipyw.interact(toto, x=qs);
```

Or equivalently using the decorator notation

```python
# equivalently
@ipyw.interact(x=qs)
def toto(x):
    return str(x*2)
```

## boxing
The `QuantityText` can wrapped in any `ipywidgets.Box` : 

```python
# define widget
qs = QuantityText(2*m)

# wrap widget in VBox
ipyw.VBox([qs, qs])
```

## interactive
The `QuantityText` can used with the `ipywidgets.interactive` decorator. The interactive widget returns a widget containing the result of the function in w.result.

```python
# interact without abbreviation
qs = QuantityText(2*m)

# define function


def slow_function(i):
    """
    Sleep for 1 second then print the argument
    """
    from time import sleep
    print('Sleeping...')
    sleep(1)
    print(i)
    print(2*i)
    return i*2


# wrap function with widget
w = ipyw.interactive(slow_function, i=qs)
```

```python
w
```

```python
w.result
```

## interact manual

```python
qs = QuantityText(2*m)


def slow_function(i):
    """
    Sleep for 1 second then print the argument
    """
    from time import sleep
    print('Sleeping...')
    sleep(1)
    print(i)


decorated_slow_function = ipyw.interact_manual(slow_function, i=qs)
```

## interactive output

Build complete UI using QuantityText for inputs, and outputs, and wrap with interactive_outputs:

```python
# inputs widgets
wa = QuantityText(2*m)
wb = QuantityText(2*m)
wc = QuantityText(2*m)

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

# define function


def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook
    # output area.
    print((a, b, c))
    res = a*b/c
    print(res)
    display(res)
    return res


# create output widget
out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})
```

```python
# display full UI
display(ui, out)
```

```python
wb.value = 12*m
```

## link
Support linking :

```python
qw1 = QuantityText(2*m)
qw2 = QuantityText(2*m)

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
```

```python
qw1
```

```python
qw2
```

## observe

```python
# create widget
qw = QuantityText(2*m)

# create text output that displays the widget value
square_display = ipyw.HTML(description="Square: ",
                           value='{}'.format(qw.value**2))

# create observe link


def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)


qw.observe(update_square_display, names='value')
```
```python
# wrap input widget and text ouput
ipyw.VBox([qw, square_display])
```
# Sliders


## Basic QuantitySlider


For simplicity purpose, the dimension cannot be changed (`w4.min = 2\*s` is not expected):

```python
from numpy import pi
import ipywidgets as ipyw
from physipy import units, m
from physipy.qwidgets.qipywidgets import QuantitySlider
mm = units["mm"]
```

```python
w4 = QuantitySlider(3*m)
w4
```

### Working with favunits
By default, anytime a Quantity is passed with a favunit, it will be used to display

```python
q = pi*m
q.favunit = mm

w6 = QuantitySlider(q)
w6
```

### Disable label

```python
w6 = QuantitySlider(q, label=False)
w6
```

```python
print(w6.value)
```

### observing


Observing works (just not the VBoxing)

```python
slider = QuantitySlider(value=7.5*m)

# Create non-editable text area to display square of value
square_display = ipyw.HTML(description="Square: ",
                           value='{}'.format(slider.value**2))

# Create function to update square_display's value when slider changes


def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)


slider.observe(update_square_display, names='value')

# Put them in a vertical box
display(slider, square_display)
```

### boxing

```python
ipyw.VBox([slider, slider])
```

### interactive output

```python
# inputs widgets
wa = QuantitySlider(2*m)
wb = QuantitySlider(2*m)
wc = QuantitySlider(2*m)

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

# define function


def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook
    # output area.
    #print((a, b, c))
    res = a*b/c
    # print(res)
    display(res)
    return res


# create output widget
out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
```

## QuantityTextSlider

```python
from physipy.qwidgets.qipywidgets import QuantityTextSlider
import ipywidgets as ipyw
```

```python
from numpy import pi
from physipy import m, units, rad
mm = units["mm"]
km = units["km"]


w = QuantityTextSlider(3*m)
w
```

```python
w = QuantityTextSlider(3*rad, min=3*rad, max = 100*rad)
w
```

```python
q = pi*m
q.favunit = mm
w = QuantityTextSlider(q)
w
```

```python
# play around with the above widget's favunit
#w.favunit = km
#w.qslider.favunit = mm
```

```python
q = pi*m
w = QuantityTextSlider(q, favunit=km)
w
```

```python
qw1 = QuantityTextSlider(2*m)
qw2 = QuantityTextSlider(2*m)

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
display(qw1, qw2)
```

```python
slider = QuantityTextSlider(
    7.5*m)

# Create non-editable text area to display square of value
square_display = ipyw.HTML(description="Square: ",
                           value='{}'.format(slider.value**2))

# Create function to update square_display's value when slider changes


def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)


slider.observe(update_square_display, names='value')

# Put them in a vertical box
display(slider, square_display)
```

```python
ipyw.VBox([slider, slider])
```

```python
# inputs widgets
wa = QuantityTextSlider(2*m, description="Toto:")
wb = QuantityTextSlider(2*m)
wc = QuantityTextSlider(2*m)

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

# define function


def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook
    # output area.
    #print((a, b, c))
    res = a*b/c
    # print(res)
    display(res)
    return res


# create output widget
out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
```

```python

```

# QuantityRangeSlider

```python
from physipy.qwidgets.qipywidgets import QuantityRangeSlider
import ipywidgets as ipyw
```

```python
from physipy import m

w = QuantityRangeSlider(3*m, 10*m, label=True)
w
```

```python
w = QuantityRangeSlider(3*m, 10*m, label=True, description="Toto")
w
```

```python
qw1 = QuantityRangeSlider(3*m, 10*m, label=True)
qw2 = QuantityRangeSlider(3*m, 10*m, label=True)

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
display(qw1, qw2)
```

```python
qw1.value
```

```python
qw2.value
```

```python
slider = QuantityRangeSlider(
    min=3*m, max=12*m)

# Create non-editable text area to display square of value
square_display = ipyw.HTML(description="Square: ",
                           value='{}-{}'.format(slider.value[0], slider.value[1]))

# Create function to update square_display's value when slider changes


def update_square_display(change):
    square_display.value = '{}-{}'.format(change.new[0]**2, change.new[1]**2)


slider.observe(update_square_display, names='value')

# Put them in a vertical box
display(slider, square_display)
```

```python
ipyw.VBox([slider, slider])
```

```python
# inputs widgets
wa = QuantityRangeSlider(min=3*m, max=12*m)
wb = QuantityRangeSlider(min=3*m, max=12*m)
wc = QuantityRangeSlider(min=3*m, max=12*m)

# An HBox lays out its children horizontally
ui = ipyw.VBox([wa, wb, wc])

# define function


def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook
    # output area.
    #print((a, b, c))

    res = a[0]*a[1]*b[0]*b[1]/c[0]*c[1]
    # print(res)
    display(res)
    return res


# create output widget
out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)
```

Favunit

```python
import physipy
mm = physipy.units["mm"]
```

```python
qw1 = QuantityRangeSlider(3*m, 10*m, label=True, favunit=mm)
qw1
```

# FavunitDropdown

```python
from physipy.qwidgets.qipywidgets import FavunitDropdown
import ipywidgets as ipyw
```

```python
w = FavunitDropdown()
w
```

```python
print(w.value)
```

```python
favunit = FavunitDropdown()

# Create non-editable text area to display square of value
favunit_display = ipyw.HTML(description="Favunit: ",
                            value='{}'.format(favunit.value))

# Create function to update square_display's value when slider changes


def update_display(change):
    favunit_display.value = '{}, as "{}"'.format(change.new, change.new.symbol)


favunit.observe(update_display, names='value')

# Put them in a vertical box
display(favunit, favunit_display)
```

```python
qw1 = FavunitDropdown()
qw2 = FavunitDropdown()

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
display(qw1, qw2)
```

```python
print(qw1.value, qw2.value)
```

```python

```

# Ideas


Implement multiple rangesliders

```python
from ipywidgets import widgets
from IPython.display import display, clear_output


def range_elems(first, last, step):
    """
    Return a list of elements starting with first, ending at last with
    stepsize of step
    """
    ret = [first]
    nxt = first + step
    while nxt <= last:
        ret.append(nxt)
        nxt += step
    return ret


class MultiRangeSlider(object):
    def __init__(self, min=0, max=1, step=0.1, description="MultiRange", disabled=False):
        self.min = min
        self.max = max
        self.step = step
        self.description = description
        self.disabled = disabled

        self.range_slider_list = []

        self.add_range_button = widgets.Button(description="Add range")
        self.add_range_button.on_click(self.handle_add_range_event)

        self.rm_range_button = widgets.Button(description="Rm range")
        self.rm_range_button.on_click(self.handle_rm_range_event)

        # combined range over all sliders, excluding possible overlaps
        self.selected_values = []

        # Vertical box for displaying all the sliders
        self.vbox = widgets.VBox()

        # create a first slider and update vbox children for displaying
        self.handle_add_range_event()

        # callback function to be called when the widgets value changes
        # this needs to accept the usual 'change' dict as in other widgets
        self.observe_callback = None

    def update_selected_values(self, change):
        """
        find the unique range points from looking at all slider ranges,
        effectively ignores overlapping areas.
        Called on every change of a single slider
        """
        range_points_lst = []
        for slider in self.range_slider_list:
            # get the current range delimiters
            r_min, r_max = slider.value
            # make sure that the range includes the endpoint r_max
            range_points = range_elems(r_min, r_max, slider.step)
            range_points_lst.append(range_points)

        # now collapse the list of lists
        flattened_range_point_lst = [
            item for lst in range_points_lst for item in lst]
        # make deep copy for callback reference
        old = [val for val in self.selected_values]
        # get unique values only
        self.selected_values = sorted(list(set(flattened_range_point_lst)))

        #print("updated self.selected_values = ", self.selected_values)
        # call the callback function if there is one
        if self.observe_callback:
            change = dict()
            change["owner"] = self
            change["type"] = "change"
            change["name"] = "value"
            change["old"] = old
            change["new"] = self.selected_values
            self.observe_callback(change)

    def handle_rm_range_event(self, b=None):
        """
        """
        if len(self.range_slider_list) > 1:
            # remove last slider
            self.range_slider_list.pop()

        # update the display
        first_line = widgets.HBox(
            [self.range_slider_list[0], self.add_range_button, self.rm_range_button])
        self.vbox.children = [first_line] + self.range_slider_list[1:]

        # update visibility of rm button
        self.rm_range_button.disabled = True if len(
            self.range_slider_list) == 1 else False

    def handle_add_range_event(self, b=None):
        """
        Callback function of the 'Add' button that displays another RangeSlider.
        """
        # adds a range slider to the list
        self.add_range_slider()
        # update elements of the displayed vbox so display will update immediately
        first_line = widgets.HBox(
            [self.range_slider_list[0], self.add_range_button, self.rm_range_button])
        self.vbox.children = [first_line] + self.range_slider_list[1:]

        # activate rm button if there is more than one slider
        self.rm_range_button.disabled = True if len(
            self.range_slider_list) == 1 else False

    def add_range_slider(self):
        """
        Add another range slider to the list of range sliders, that, when its
        value changes, updates the combined range of the current object.
        """
        # a new range slider is requested, but don't show description again
        slider = widgets.FloatRangeSlider(
            min=self.min,
            max=self.max,
            step=self.step,
            disabled=self.disabled,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        if not self.range_slider_list:
            # first slider gets a description
            slider.description = "MultiRangeSlider"

        # when its value changes, update internal selection of combined range
        slider.observe(self.update_selected_values, names='value')

        self.range_slider_list.append(slider)

    def display(self):
        """
        Show the widget in the notebook.
        """
        # create a vbox that contains all sliders below each other
        display(self.vbox)

    def observe(self, fun):
        """
        Set the callback function that is called when any of the RangeSliders changes
        """
        self.observe_callback = fun
```

```python
multi_range = MultiRangeSlider()
multi_range.observe(lambda change: print(change["new"]))
multi_range.display()
```

```python
from ipywidgets import interact, interactive


def f(x):
    return 2*x


w = interact(f, x=10);
```

```python
w(2)
```

```python
w = interactive(f, x=10)
```

```python
w
```

```python
print(w.result)
```

```python
w
```

```python
w = interactive(f, x=ipyw.FloatSlider(min=30, max=40))
```

```python
w
```

```python
w.result
```

```python
type(w)
```

```python
a = ipyw.IntSlider()
b = ipyw.IntSlider()
c = ipyw.IntSlider()
ui = ipyw.HBox([a, b, c])


def f(a, b, c):
    return a*b*c


out = ipyw.interactive_output(f, {'a': a, 'b': b, 'c': c})
```

```python
out
```

```python

```

```python

```
