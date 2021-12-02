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
from physipy import random as phyrandom, s
import numpy as np

# %%
np.random.normal(1, 2, 10000)*s

# %%
print(phyrandom.normal(1, 2, 10000))
print(phyrandom.normal(1, 2, 10000)*s)
print(phyrandom.normal(1*s, 2*s, 10000))

# %%
