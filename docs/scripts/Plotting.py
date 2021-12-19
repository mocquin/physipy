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
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib

from physipy import Dimension, units, quantify, Quantity

ms = units["ms"]
mm = units['mm']
km = units["km"]
cm = units["cm"]


# %% [markdown]
# Ressources :
#  - https://github.com/matplotlib/matplotlib/blob/97115aef5c18af5e48eb4ef041b6f48567088874/lib/matplotlib/axis.py#L1521
#  
# print(ax.xaxis.have_units())
# print(ax.yaxis.have_units()) 
# print(ax.xaxis.have_units())
# print(ax.xaxis.converter)
# print(ax.xaxis.units)
# print(ax.xaxis.get_units())
# print(ax.xaxis.set_units("totot"))
#
# - See astropy for plotting context : https://docs.astropy.org/en/stable/_modules/astropy/visualization/units.html#quantity_support
# - Astropy known issues : https://docs.astropy.org/en/stable/known_issues.html#quantity-issues
#
# - artist
# https://github.com/matplotlib/matplotlib/blob/87119ea07357bc065bf729bfb7cd35e16dffe91b/lib/matplotlib/artist.py#L188

# %% [markdown]
# # Plotting with matplotlib

# %% [markdown]
# By default, Quantity' are plotted with their raw value, ie si-unit value

# %%
y = np.linspace(0, 30) * mm
x = np.linspace(0, 5) * s

fig, ax = plt.subplots()
ax.plot(x, y, 'tab:blue')
ax.plot(3*x-2*s, 3*y+3*mm)

# %% [markdown]
# # Plotting with matplotlib in a context

# %% [markdown]
# Using a context to only use the Quantity interface for plotting :

# %%
with physipy.quantity.plot.plotting_context():
    y = np.linspace(0, 30) * mm
    x = np.linspace(0, 5) * s
    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'tab:blue')
    ax.axhline(0.02 * m, color='tab:red')
    ax.axvline(500*ms, color='tab:green')

# %% [markdown]
# Then outside the context the behaviour is the same as by default, ie without calling `setup_matplotlib()`:

# %%
y = np.linspace(0, 30) * mm
x = np.linspace(0, 5) * s

fig, ax = plt.subplots()
ax.plot(x, y, 'tab:blue')

# %% [markdown]
# # Quick-Plotting shortcut

# %% [markdown]
# Quick plot an array-like quantity

# %%
x.plot()

# %% [markdown]
# # Plotting with matplotlib

# %% [markdown]
# Examples taken from [pint](https://pint.readthedocs.io/en/stable/plotting.html).
# Make sure you enable units handling in matplotlib with `setup_matplotlib`

# %%
setup_matplotlib()

y = np.linspace(0, 30) * mm
x = np.linspace(0, 5) * s

fig, ax = plt.subplots()
#ax.plot(x, y, 'tab:blue')
ax.axhline(0.02 * m, color='tab:red')
ax.axvline(500*ms, color='tab:green')

# %%
import matplotlib.pyplot as plt
import numpy as np

y = np.linspace(0, 30) * mm
x = np.linspace(0, 5) * s

fig, ax = plt.subplots()
ax.yaxis.set_units(mm)
ax.xaxis.set_units(ms)

ax.plot(x, y, 'tab:blue')
ax.axhline(0.02 * m, color='tab:red')
ax.axvline(500*ms, color='tab:green')

# %% [markdown]
# The axis units can be changed after the values are plotted as wellimport matplotlib.pyplot as plt
# import numpy as np

# %%
y = np.linspace(0, 30) * mm
x = np.linspace(0, 5) * ms

fig, ax = plt.subplots()
ax.plot(x, y, 'tab:blue')
ax.axhline(26400 * mm, color='tab:red')
ax.axvline(120 * ms, color='tab:green')
ax.yaxis.set_units(mm)
ax.xaxis.set_units(ms)
ax.autoscale_view()

# %% [markdown]
# # Plotting with favunit

# %% [markdown]
# If the Quantity objects that are called in `ax.plot` have favunit, it will be used by default as the axis's unit.

# %%
y = np.linspace(0, 30) * mm
x = np.linspace(0, 5) * s
y.favunit = mm # no need to call ax.yaxis.set_units(mm)
x.favunit = ms # no need to call ax.xaxis.set_units(ms)

fig, ax = plt.subplots()
ax.plot(x, y, 'tab:blue')
ax.axhline(0.02 * m, color='tab:red')
ax.axvline(500*ms, color='tab:green')

# %% [markdown]
# # Known issues

# %% [markdown]
# ## axvline and friends

# %% [markdown]
# Without units implemented, the `axvline` and friend use a comparison to the axis values which are by default floats. Hence the comparaison with a quantity that have a dimension fails. It works with a dimensionless Quantity because of quantify turns the axis values to Quanities, and the comparison works

# %%
# restart kernel
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib
from physipy import Dimension, units, quantify, Quantity


ms = units["ms"]
mm = units['mm']
km = units["km"]
cm = units["cm"]


# %% [markdown]
# Without setup : 

# %%
try:
    plt.axvline(3*s)
except Exception as e:
    print(e)

# %% [markdown]
# Without setup but plotting a dimensionless Quantity

# %%
plt.axvline(Quantity(3, Dimension(None)))

# %% [markdown]
# Works when calling setup_matplotlib() : 

# %%
setup_matplotlib()
try:
    plt.axvline(3*s)
except Exception as e:
    print(e)

# %%
