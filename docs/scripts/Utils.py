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
# # Utils
# In addition to the decorators, some utilities are available.

# %% [markdown]
# ## numpy's arange equivalent

# %%
from physipy import m
from physipy.quantity.utils import qarange
import numpy as np

# %%
qarange(1*m, 10*m, 0.5*m)

# %% [markdown]
# ## convert arrays to Quantity

# %% [markdown]
# Turn array of Quantity's to Quantity with array-value

# %%
from physipy.quantity.utils import asqarray

# %%
arr_of_Q = [m, 2*m, 3*m]
print(arr_of_Q)
print(asqarray(arr_of_Q))

# %% [markdown]
# Normal array will be turned to quantity

# %%
dimless = asqarray(np.array([1, 2, 3]))
print(dimless)
print(type(dimless))

# %%
dimless

# %%
