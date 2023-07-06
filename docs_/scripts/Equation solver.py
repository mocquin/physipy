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

# %% [markdown]
# # Using scipy wrappers

# %% [markdown]
# Scipy offers various solver algorithms in `scipy.optimize`. Some of the solvers are wrapped and presented below.

# %% [markdown]
# ## Root solver

# %% [markdown]
# A wrapper of `scipy.optimize.root`:

# %%
from physipy import s
from physipy.optimize import root

def toto(t):
    return -10*s + t


# %%
print(root(toto, 0*s))


# %%
def tata(t, p):
    return -10*s*p + t

print(root(tata, 0*s, args=(0.5,)))

# %% [markdown]
# ### Quadratic Brent method

# %% [markdown]
# A wrapper of `scipy.optimize.brentq`:

# %%
from physipy.optimize import brentq


# %%
print(brentq(toto, -10*s, 10*s))
print(brentq(tata, -10*s, 10*s, args=(0.5,)))


# %%
