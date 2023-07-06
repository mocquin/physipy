# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import physipy
from physipy import m, s, Quantity, Dimension, rad, units
from physipy.quantity.utils import *


# %% [markdown]
# # Usefull decorators for dimensionfullness functions 

# %% [markdown]
# Plenty of decorators are available.

# %% [markdown]
# ## Basic dimension checking
# The first decorator simply checks the dimension of the inputs and/or outputs.
# This can be used to :
#  - avoid timy calculation that end up on a DimensionError
#  - check the dimension of the output at its creation, not later when used with other quantities
#  - quickly check that the function you implemented returns the expected dimension
#  - restrict a function use to a specific dimension
#
# To specify the dimension:
#  - a quantity can be used
#  - a string represnetating the dimension,  like "L"
#  - a Dimension object

# %%
@check_dimension(("L", "L"), ("L"))
def sum_length(x, y):
    return x+y+1*m

print(sum_length(1*m, 1*m))


@check_dimension((m, m), (m))
def sum_length(x, y):
    "This function could be used on any Quantity, but here restricted to lengths."
    return x+y

print(sum_length(1*m, 1*m))


@check_dimension((Dimension("L"), Dimension("L")), (Dimension("L")))
def sum_length(x, y):
    return x+y

print(sum_length(1*m, 1*m))


# %% [markdown]
# ## Favunit setting
# This decorator simply sets the favunit of the outputs

# %%
mm = units["mm"]

@set_favunit(mm)
def sum_length(x, y):
    return x+y+1*m
print(sum_length(1*m, 1.123*m))


# %% [markdown]
# ## Dimension checks and favunit setting

# %% [markdown]
# This decorator is a wrapper on `set_favunit` and `check_dimension`. The outputs' object will be used to check dimension and set as favunit

# %%
@dimension_and_favunit((m, m), mm)
def sum_length(x, y):
    return x+y+1*m
print(sum_length(1*m, 1.123*m))


# %% [markdown]
# ## Convert quantitys to dimensionless quantities
# Wrap functions that expect floats value in a certain unit

# %%
@convert_to_unit(mm, mm)
def sum_length_from_floats(x_mm, y_mm):
    """Expects values as floats in mm"""
    return x_mm + y_mm + 1
print(sum_length_from_floats(1.2*m, 2*m))


# %% [markdown]
# ## Drop dimension
# Send the si value

# %%
@drop_dimension
def sum_length_from_floats(x, y):
    """Expect dimensionless objects"""
    return x + y + 1
print(sum_length_from_floats(1.2*m, 2*m))


# %% [markdown]
# ## Adding units to ouputs

# %%
@add_back_unit_param(m, s)
def timed_sum(x_m, y_m):
    time = 10
    return x_m + y_m + 1, time
print(timed_sum(1, 2))


# %% [markdown]
# ## Enforce consistents dimension
# Force same dimension for inputs, without specifying which dimension

# %%
@decorate_with_various_unit(('A', 'A'), 'A')
def another_sum(x, y):
    return x + y
print(another_sum(2*m, 1*m))
