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
# # Dimension

# %% [markdown]
# Sympy is dimension is only used for printing and parsing. 

# %% [markdown]
# # 

# %%
from physipy import m
import numpy as np

# %%

# %%
from physipy import m, units
mm = units["mm"]

# %%
(mm**2).symbol

# %%
res = np.linspace(0, 10)*m
res.symbol

# %%

print((np.linspace(0, 10)*m).symbol)
print((np.linspace(0, 10)/m).symbol)
print((m*m).symbol)
print((m**2).symbol)
print((2*m).symbol)
print((2*m**2).symbol)
print((mm**2).symbol)
print((2*mm).symbol)


# %%
str((2*mm).symbol)

# %%
