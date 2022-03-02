# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Idea

# %% [markdown]
# Often I was ending up defining an interface to handle a quantity using both a widget, mostly sliders, and as an attribute, like this : this allows to use both a slider and an attribute to control the value of a quantity

# %%
from physipy import m, units, s
from physipy.qwidgets.qipywidgets import QuantitySlider, QuantityTextSlider
W = units["W"]

class Car():
    def __init__(self, max_speed, power):
        # define widgets
        self.max_speed_w = QuantitySlider(min=0*m/s, max=100*m/s, value=max_speed, description="Max speed")
        self.power_w = QuantitySlider(min=0*W, max=100*W, value=power, description="Power")
        
    @property
    def max_speed(self):
        return self.max_speed_w.value
    @property
    def power(self):
        return self.power_w.value
    @max_speed.setter
    def max_speed(self, value):
        self.max_speed_w.value = value
    @power.setter
    def power(self, value):
        self.power_w.value = value
        
        
car = Car(30*m/s, 200*W)

print(car.power)
display(car.power)
display(car.power_w)

# %%
print(car.power)


# %% [markdown]
# But notice that a lot of code is repetitive and not really interesting, it just define an interface and relation between the slider and a property-like attribute.
# This is ok for a class with 2 sliders, but it quickly get messy for many quantities. 

# %% [markdown]
# That's where descriptors come to the rescue.

# %% [markdown]
# Now the same Car class can be concisely writtern : 

# %%
from physipy import m, s, units
from physipy.qwidgets.qipywidgets import QuantitySliderDescriptor
W = units["W"]


class Car():
    max_speed = QuantitySliderDescriptor(min=0*m/s, max=100*m/s)
    power = QuantitySliderDescriptor(min=0*W, max=100*W)
    
    def __init__(self, max_speed, power):
        self.max_speed = max_speed
        self.power = power
        
car = Car(30*m/s, 200*W)
print(car.power)
display(car.power)
display(car.power_w)

# %%
display(car.power)
display(car.power_w)

# %%
