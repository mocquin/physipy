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
# # Repr and string

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import physipy
from physipy import m, s, Quantity, Dimension, rad

# %% [markdown]
# ## Interactive repr

# %%
from physipy import K, constants
sigma = constants["sigma"]


# %%
sigma * 300*K

# %%
