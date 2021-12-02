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
import pint
import numpy as np

# %%
ureg = pint.UnitRegistry()

# %%
np.full((3, 3), 2*ureg.m)

# %%
a = np.array([1, 2, 3]) * ureg.m
b = np.array([1, 1, 1]) * ureg.m
np.copyto(a, b)
print(a)
print(b)

# %%
a = np.array([1, 2, 3]) * ureg.m
b = np.array([1, 1, 1]) 
np.copyto(a, b)
print(a)
print(b)

# %%
a = np.array([1, 2, 3])
b = np.array([1, 1, 1]) * ureg.m
np.copyto(a, b)
print(a)
print(b)

# %% [markdown]
# # from array(2m)

# %%
np.array(2*ureg.m)

# %%
