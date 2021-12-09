# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from physipy.qwidgets.qipywidgets import QuantityText, FDQuantityText, QuantityTextSlider, FavunitDropdown
import ipywidgets as ipyw
from numpy import pi
from physipy import m, s, Quantity, Dimension, rad, units

mm = units["mm"]

# %%
from physipy import m, units
mm = units["mm"]

a = 3*m
a.favunit = mm
w = QuantityTextSlider(a)
w

# %%
w = FavunitDropdown()
w

# %% [markdown]
# TODO : 
#  - ajouter des min/max/step au QuantityTextSlider

# %% [markdown]
# Things to test :
#
# Types of widgets :
#  - QuantityText : text area, very permissive
#  - FDQuantityText : text area, fixed dimension on init value
#  - FDQuantitySlider : slider, fixed dimension on init value
#  - FDQuantitySliderWithBounds : slider with FDQuantityText to handle min and max
#  
# Functionnalitities : 
#  - VBoxing, HBoxing : `ipyw.VBox([qw, qw])`
#  - interact with abbreviation : `interact(3*m)`
#  - interact with widget : `interact(QuantityText(3*m))`
#  - interactive : `w = ipyw.interactive(slow_function, i=qs)`
#  - interactive_output : `out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})`
#  - observe : 
#  - link : `mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))`
#  - jslink : `mylink = ipyw.jslink((qw1, 'value'), (qw2, 'value'))`

# %%
q = 70*mm
q.favunit = mm

# %% [markdown]
# # Text

# %% [markdown]
# 2 types that inherit from QuantityText : 
#  - free QuantityText
#  - Fixed-dimension "FDQuantityText"

# %% [markdown]
# ## QuantityText
# Basically a text area that will parse python expression into a numerical physical Quantity. It can be dimensionless (like 5 or 2\*pi), and can be of any dimension at any time:

# %%
w = QuantityText()
w

# %% [markdown]
# A QuantityText has typical attributes of a widget and a Quantity, except the `value` is the actual Quantity, not the value of the Quantity : 

# %%
print(w.value)
print(w.value.value)
print(w.dimension)
print(w.description)
print(w.fixed_dimension)

# %%
print(w.value)
print(w.dimension)
print(w.description)

# %%
print(w.value)
print(w.dimension)
print(w.description)

# %% [markdown]
# With custom description : 

# %%
QuantityText(description="Weight")

# %% [markdown]
# With init value

# %%
w = QuantityText(2*pi*s)
w

# %%
QuantityText(2*pi*rad, description="Angle")

# %% [markdown]
# A `fixed_dimension` attribute can be set to allow change of dimension. By default is false, and so dimension can be changed

# %%
# start with seconds ...
a = QuantityText(2*pi*s)
print(a.fixed_dimension)
a

# %%
# ... then change into radians
a.value = 2*rad
print(a.value)
print(a.fixed_dimension)

# %%
# if fixed_dimension=True ...
b = QuantityText(2*pi*s, fixed_dimension=True)
b

# %%
# ... cannot be changed
try:
    b.value = 2*m
except:
    print("b.fixed_dimension =", b.fixed_dimension,
          ", hence Quantity must be same dimension.")

# %%
# handle favunit
a = 3*m
a.favunit = mm
w = QuantityTextSlider(a)
w

# %%
w = QuantityTextSlider(3*m)
w

# %%
w.qslider.favunit = mm
w

# %% [markdown]
# # Fixed-Dimension QuantityText
# A QuantityText that will set a dimension at creation, and not allow any other dimension:

# %% [markdown]
# A fixed-dimensionless quantity : (trying to set a quantity with another dimension will be ignored : example : type in '5\*m' then Enter)

# %%
# init at value 0, then change its value and print
w2 = FDQuantityText()
w2

# %%
print(w2.value)

# %% [markdown]
# A fixed-length quantity :

# %%
# create with a length, then only another can be set
w3 = FDQuantityText(pi*m)
w3

# %%
print(w3.value)

# %% [markdown]
# # Testing with QuantityText

# %% [markdown]
# ## interact without abbreviation

# %% [markdown]
# The `QuantityText` widget can be passed to `ipywidgets.interact` decorator : 

# %%
# define widget
qs = QuantityText(2*m)

# define function


def toto(x):
    return str(x*2)


# wrap function with interact and pass widget
res = ipyw.interact(toto, x=qs);


# %% [markdown]
# Or equivalently using the decorator notation

# %%
# equivalently
@ipyw.interact(x=qs)
def toto(x):
    return str(x*2)


# %% [markdown]
# ## boxing
# The `QuantityText` can wrapped in any `ipywidgets.Box` : 

# %%
# define widget
qs = QuantityText(2*m)

# wrap widget in VBox
ipyw.VBox([qs, qs])

# %% [markdown]
# ## interactive
# The `QuantityText` can used with the `ipywidgets.interactive` decorator. The interactive widget returns a widget containing the result of the function in w.result.

# %%
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

# %%
w

# %%
w.result

# %% [markdown]
# ## interact manual

# %%
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

# %% [markdown]
# ## interactive output
#
# Build complete UI using QuantityText for inputs, and outputs, and wrap with interactive_outputs:

# %%
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

# %%
# display full UI
display(ui, out)

# %%
wb.value = 12*m

# %% [markdown]
# ## link
# Support linking :

# %%
qw1 = QuantityText(2*m)
qw2 = QuantityText(2*m)

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))

# %%
qw1

# %%
qw2

# %% [markdown]
# ## observe

# %%
# create widget
qw = QuantityText(2*m)

# create text output that displays the widget value
square_display = ipyw.HTML(description="Square: ",
                           value='{}'.format(qw.value**2))

# create observe link


def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)


qw.observe(update_square_display, names='value')
# %%
# wrap input widget and text ouput
ipyw.VBox([qw, square_display])
# %% [markdown]
# # Sliders

# %% [markdown]
# ## Basic QuantitySlider

# %% [markdown]
# For simplicity purpose, the dimension cannot be changed (`w4.min = 2\*s` is not expected):

# %%
from physipy.qwidgets.qipywidgets import QuantitySlider

# %%
w4 = QuantitySlider(3*m)
w4

# %% [markdown]
# ### Working with favunits
# By default, anytime a Quantity is passed with a favunit, it will be used to display

# %%
q = pi*m
q.favunit = mm

w6 = QuantitySlider(q)
w6

# %% [markdown]
# ### Disable label

# %%
w6 = QuantitySlider(q, label=False)
w6

# %%
print(w6.value)

# %% [markdown]
# ### observing

# %% [markdown]
# Observing works (just not the VBoxing)

# %%
slider = QuantitySlider(
    value=7.5*m)

# Create non-editable text area to display square of value
square_display = ipyw.HTML(description="Square: ",
                           value='{}'.format(slider.value**2))

# Create function to update square_display's value when slider changes


def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)


slider.observe(update_square_display, names='value')

# Put them in a vertical box
display(slider, square_display)

# %% [markdown]
# ### boxing

# %%
ipyw.VBox([slider, slider])

# %% [markdown]
# ### interactive output

# %%
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

# %%

# %%

# %%

# %% [markdown]
# ## QuantityTextSlider

# %%
from physipy.qwidgets.qipywidgets import QuantityTextSlider
import ipywidgets as ipyw

# %%
from numpy import pi
from physipy import m, units, rad
mm = units["mm"]
km = units["km"]


w = QuantityTextSlider(3*m)
w

# %%
w = QuantityTextSlider(3*rad, min=3*rad, max = 100*rad)
w

# %%
q = pi*m
q.favunit = mm
w = QuantityTextSlider(q)
w

# %%
# play around with the above widget's favunit
#w.favunit = km
#w.qslider.favunit = mm

# %%
q = pi*m
w = QuantityTextSlider(q, favunit=km)
w

# %%
qw1 = QuantityTextSlider(2*m)
qw2 = QuantityTextSlider(2*m)

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
display(qw1, qw2)

# %%
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

# %%
ipyw.VBox([slider, slider])

# %%
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

# %%

# %% [markdown]
# # QuantityRangeSlider

# %%
from physipy.qwidgets.qipywidgets import QuantityRangeSlider
import ipywidgets as ipyw

# %%
from physipy import m

w = QuantityRangeSlider(3*m, 10*m, label=True)
w

# %%
w = QuantityRangeSlider(3*m, 10*m, label=True, description="Toto")
w

# %%
qw1 = QuantityRangeSlider(3*m, 10*m, label=True)
qw2 = QuantityRangeSlider(3*m, 10*m, label=True)

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
display(qw1, qw2)

# %%
qw1.value

# %%
qw2.value

# %%
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

# %%
ipyw.VBox([slider, slider])

# %%
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

# %% [markdown]
# Favunit

# %%
import physipy
mm = physipy.units["mm"]

# %%
qw1 = QuantityRangeSlider(3*m, 10*m, label=True, favunit=mm)
qw1

# %% [markdown]
# # FavunitDropdown

# %%
from physipy.qwidgets.qipywidgets import FavunitDropdown
import ipywidgets as ipyw

# %%
w = FavunitDropdown()
w

# %%
print(w.value)

# %%
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

# %%
qw1 = FavunitDropdown()
qw2 = FavunitDropdown()

# create link
mylink = ipyw.link((qw1, 'value'), (qw2, 'value'))
display(qw1, qw2)

# %%
print(qw1.value, qw2.value)

# %%

# %% [markdown]
# # Ideas

# %% [markdown]
# Implement multiple rangesliders

# %%
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


# %%
multi_range = MultiRangeSlider()
multi_range.observe(lambda change: print(change["new"]))
multi_range.display()

# %%
from ipywidgets import interact, interactive


def f(x):
    return 2*x


w = interact(f, x=10);

# %%
w(2)

# %%
w = interactive(f, x=10)

# %%
w

# %%
print(w.result)

# %%
w

# %%
w = interactive(f, x=ipyw.FloatSlider(min=30, max=40))

# %%
w

# %%
w.result

# %%
type(w)

# %%
a = ipyw.IntSlider()
b = ipyw.IntSlider()
c = ipyw.IntSlider()
ui = ipyw.HBox([a, b, c])


def f(a, b, c):
    return a*b*c


out = ipyw.interactive_output(f, {'a': a, 'b': b, 'c': c})

# %%
out

# %%

# %%
