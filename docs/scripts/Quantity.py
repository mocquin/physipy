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

# %% [markdown]
# # Quantity

# %% [markdown]
# ## Construction from class

# %% [markdown]
# Several ways are available constructing a Quantity object

# %%
from physipy import Quantity, m, sr, Dimension

# %% [markdown]
# Scalar Quantity

# %%
# from int
print(Quantity(1, Dimension("L")))

# from float
print(Quantity(1.123, Dimension("L")))

# from fraction.Fraction
from fractions import Fraction
print(Quantity(Fraction(1, 2), Dimension("L")))


# %% [markdown]
# Array-like Quantity

# %%
# from list
print(Quantity([1, 2, 3, 4], Dimension("L")))

# from tuple
print(Quantity((1, 2, 3, 4), Dimension("L")))

# from np.ndarray
import numpy as np
print(Quantity(np.array([1, 2, 3, 4]), Dimension("L")))

# %% [markdown]
# ## Construction by multiplicating value with unit/quantity

# %% [markdown]
# Because the multiplication of quantity first tries to "quantify" the other operand, several creation routines by multicpliation are available

# %%
# multiplicating int
print(1 * m)

# multiplicating float
print(1.123 * m)

# multiplicating Fraction
print(Fraction(1, 2) * m)

# multiplicating list
print([1, 2, 3, 4] * m)

# multiplicating list
print((1, 2, 3, 4) * m)

# multiplicating array
print(np.array([1, 2, 3, 4]) * m)

# %% [markdown]
# # Known issues

# %% [markdown]
# ## Quantity defition with minus sign

# %%
from physipy import Quantity, Dimension

print(type(-Quantity(5, Dimension(None)))) # returns int
print(type(Quantity(-5, Dimension(None)))) # returns Quantity
print(type(Quantity(5, Dimension(None)))) # returns Quantity

# %%
