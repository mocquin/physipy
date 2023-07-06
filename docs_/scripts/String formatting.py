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
# # String formatting

# %%
import physipy
from physipy import m

# %%
q = 1.23456789*m

# %%
print(str(q))

# %%
q._compute_value()

# %% [markdown]
# ## Standard formatting

# %%
print(f"{q:.2f}")
print(f"{q:+.2f}")
print(f"{q:+9.2f}")
print(f"{q:*^15}")
print(f"{q: >-12.3f}")

# %% [markdown]
# ## physipy formatting

# %%
print(f"{q}")
print(f"{q:~}")  # ~ : no prefix before unit


# %%

# %%
