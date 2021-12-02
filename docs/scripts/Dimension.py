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
import physipy
from physipy import Dimension
from fractions import Fraction

# %% [markdown]
# # Dimension

# %% [markdown]
# ## Dimension's dict

# %% [markdown]
# The Dimension class is based on a single dictionnary, stored as a json file "`dimension.txt`":

# %%
for key, value in physipy.quantity.dimension.SI_UNIT_SYMBOL.items():
    print(f"{key: >5} : {value: <3}")

# %% [markdown]
# ## Construction

# %% [markdown]
# The Dimension object is basically a dictionnary that stores the dimensions' name and power. A dimension can be created different ways. The values associated can be int, float, or fractions.Fraction (actually, anything that supports addition, subtraction, multiplication, "minus" notation, and can be parsed by sympy). If possible, the values are casted into integers.

# %% [markdown]
#  - from None to create dimensionless

# %%
dimensionless = physipy.Dimension(None)
print(dimensionless)
print(repr(dimensionless))
dimensionless

# %% [markdown]
#  - from a string of a single dimension

# %%
a_length_dimension = physipy.Dimension("L")
print(a_length_dimension)
print(repr(a_length_dimension))
a_length_dimension

# %%
Dimension({"L":Fraction(1/2)})

# %% [markdown]
#  - from a string of a single dimension's SI unit symbol

# %%
a_length_dimension = physipy.Dimension("m")
print(a_length_dimension)
print(repr(a_length_dimension))
a_length_dimension

# %% [markdown]
#  - form a dict of dimension symbols

# %%
a_speed_dimension = physipy.Dimension({"L": 1, "T":-1})
print(a_speed_dimension)
print(repr(a_speed_dimension))
a_speed_dimension

# %% [markdown]
#  - from a string of a product-ratio of dimension symbols

# %%
complex_dim = physipy.Dimension("L**2/T**3*theta**(-1/2)")
print(complex_dim)
print(repr(complex_dim))
complex_dim

# %% [markdown]
#  - from a string of a product-ratio of dimension's SI unit symbols

# %%
complex_dim = physipy.Dimension("m**2/s**3*K**-1")
print(complex_dim)
print(repr(complex_dim))
complex_dim

# %% [markdown]
# ## Operations on Dimension : mul, div, pow
# Dimension implements the following :
#  - multiplication with another Dimension
#  - division by another Dimension
#  - pow by a number : this can be int, float, fractions.Fraction

# %% [markdown]
# Dimensions can be multiplied and divided together as expected : 

# %%
product_dim = a_length_dimension * a_speed_dimension
print(product_dim)
product_dim

# %%
div_dim = a_length_dimension / a_speed_dimension
print(div_dim)
div_dim

# %% [markdown]
# The inverse of a dimension can be computed by computing the division from 1, and the inverse method

# %%
1/a_speed_dimension

# %%
a_speed_dimension.inverse()

# %% [markdown]
# Computing the power : 

# %%
a_speed_dimension**2

# %%
a_speed_dimension**(1/2)

# %%
a_speed_dimension**Fraction(1/2) * a_length_dimension**Fraction(10/3)

# %% [markdown]
# ## Not implemented operations
#  - addition and substraction by anything
#  - multiplication by anything that is not a Dimension
#  - division by anaything that is not a Dimension or 1

# %%
# a_speed_dimension + a_speed_dimension --> NotImplemented
# a_speed_dimension / 1 --> TypeError: A dimension can only be divided by another dimension, not 1.
# a_speed_dimension * 1 --> TypeError: A dimension can only be multiplied by another dimension, not 1

# %% [markdown]
# ## Printing and display : str, repr, latex

# %% [markdown]
# You can display a dimension many different ways : 
#  - with the standard repr format : `repr()`
#  - as a latex form : `_repr_latex_`
#  - in terms of dimension symbol : `str`
#  - in terms of corresponding SI unit (returns a string) : `str_SI_unit()`
#
# Note that Dimension implements `__format__`, which is directly applied to its string representation.

# %%
print(complex_dim.__repr__())
print(complex_dim._repr_latex_())
print(complex_dim.__str__())
print(complex_dim.str_SI_unit())

# %% [markdown]
# In a notebook, the latex form is automaticaly called and rendered : 

# %%
complex_dim

# %% [markdown]
# ## Introspection : siunit_dict, dimensionality

# %% [markdown]
# A dict containing the SI unit symbol as keys can be accessed :

# %%
a_speed_dimension.siunit_dict()

# %% [markdown]
# A high-level "dimensionality" can be accessed : 

# %%
a_speed_dimension.dimensionality

# %% [markdown]
# The available dimensionality are stored in a dict :
#

# %%
from physipy.quantity.dimension import DIMENSIONALITY

# %%
for k, v in DIMENSIONALITY.items():
    print(f"{k: >20} : {v: >20} : {v.str_SI_unit(): >20}")
