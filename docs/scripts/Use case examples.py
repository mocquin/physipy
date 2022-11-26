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

# %%
import numpy as np
from physipy import units, m, s, K

Hz = units['Hz']

# defines variables
a = np.random.randn(10, 10) * s
b = 6.75*np.pi * 1/Hz
c = a**2 /b**0.5

np.sum(c)
