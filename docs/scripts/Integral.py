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
# # Integrals

# %% [markdown]
# There are several ways to compute integrals : 
#  - from a np.array, using the `.integrate()` that relies on `np.trapz`
#  - use `np.trapz`
#  - use `scipy.integrate.romb` or `scipy.integrate.simps`  or `scipy.integrate.trapz`
#  - use `physipy.quad`, that just wraps `scipy.integrate.quad` (or dblquad or tplquad)
#

# %%
import physipy
from physipy import m, units, s, K
import numpy as np
mm = units["mm"]

# %%
distances = np.linspace(1, 3, num=3)*m
distances

# %% [markdown]
# ## Trapezoidal rule

# %%
# computes ((1+2)/2 + (2+3)/2)
distances.integrate()

# %%
np.trapz(distances)

# %%
# use specific, constant spacing
dx = 1*s
# with float dx
print(np.trapz(distances, dx=1))
# with quantity dx
print(np.trapz(distances, dx=1*m))

# %% [markdown]
# This will work for integration of nd arrays. For example, computing several integrals : 

# %%
# sampling
ech_t = np.linspace(1, 100)*s
# params 
ech_v = np.linspace(10, 20)*m/s
Ts, Vs = np.meshgrid(ech_t, ech_v)
D = Ts*Vs
D.integrate(axis=1, x=ech_t)

# %% [markdown]
# # Trapz for 2D integral

# %%
from physipy.quantity.calculus import trapz2

# %%
#sample a 2 squared meter, in both direction with different spacing
nx = 12
ny = 30
ech_dx = np.linspace(0*m, 2*m, num=nx)
ech_dy = np.linspace(0*m, 1*m ,num=ny)
X, Y = np.meshgrid(ech_dx, ech_dy)
# make a uniform ponderation
Zs = np.ones_like(X)
print(trapz2(Zs, ech_dx, ech_dy))

# %% [markdown]
# # Scipy

# %%
import scipy

# %%
# scipy.integrate.trapz just wraps numpy's trapz
print(scipy.integrate.trapz(distances, dx=1))
print(scipy.integrate.trapz(distances, dx=1*m))

# %%
# scipy.integrate.simps : simpson's method : approximate function's interval by polynome 
# https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Simpson
scipy.integrate.simps(distances)
scipy.integrate.simps(distances, dx=1*m)

# %%
# scipy.integrate.romb : Romberg's method 
# https://en.wikipedia.org/wiki/Romberg%27s_method
scipy.integrate.romb(distances)
scipy.integrate.romb(distances, dx=1*m)


# %% [markdown]
# ## quad

# %%
def f(t):
    return t + 1*s

integ, err = physipy.quad(f, 0*s, 10*s)
integ


# %% [markdown]
# ## dblquad

# %%
def f(t, d):
    return (t + 1*s) * (d + 1*m)

integ, err = physipy.dblquad(f, 0*m, 10*m, 0*s, 10*s)
integ


# %% [markdown]
# ## tplquad

# %%
def f(t, d, deg):
    return (t + 1*s) * (d + 1*m) * (deg + 1*K)

integ, err = physipy.tplquad(f, 0*K, 10*K, 0*m, 10*m, 0*s, 10*s)
integ

# %%
