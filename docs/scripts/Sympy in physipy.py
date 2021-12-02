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
# # Dimension

# %% [markdown]
# Sympy is dimension is only used for printing and parsing. 

# %% [markdown]
# # 

# %%
from physipy import m

# %%

# %%
from physipy import m, units
mm = units["mm"]

# %%
(mm**2).symbol

# %%
