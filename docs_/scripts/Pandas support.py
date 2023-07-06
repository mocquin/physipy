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

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# # Pandas support

# %% [markdown]
# Without anything else, physipy is kinda supported in pandas, but performances will be quite degraded. It seems a 1d quantity array will be split element-wise and stored, hence all operations will be done "loop"-wise, loosing the power of numpy arrays.
#
# See [physipandas](https://github.com/mocquin/physipandas) for better interface between pandas and physipy.

# %%
import numpy as np
import pandas as pd

import physipy
from physipy import K, s, m, kg, units

# %%
arr = np.arange(10)
heights = np.random.normal(1.8, 0.1, 10)*m
heights.favunit = units["mm"]
weights = np.random.normal(74, 3, 10)*kg
heights

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Dataframe

# %%
df = pd.DataFrame({
    "heights":heights,
    "temp":arr*K,
    "weights":weights, 
    "arr":arr,
})

# %%
df["heights"]

# %%
df

# %%
df.info()

# %%
df.describe()

# %%
# %timeit df["heights"].min()

# %%
# %timeit df["arr"].min()
