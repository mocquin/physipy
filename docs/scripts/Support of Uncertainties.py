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
# # Uncertainties support
# As of now, basic operations are transparently handled.
#
# Known issues : 
#  - `x.nominal_value` will return the float nominal value, while you would like a quantity value (3 m, not 3). Same goes for `std_dev` and `std_score`
#  - `uncertainties.umath` will fail on non-dimensionless objects, but that the case also for Quantity with physipy.math
#  - there probably is a need for work on `(2*x*m+3*m).derivatives[x]` to be done
#  
# No array support testing has been done yet.
# Also, some printing/formating testing must be done.

# %%
import numpy as np
import physipy
from physipy.quantity.utils import asqarray
from physipy import m, K, s, Quantity, Dimension

# %%
import uncertainties
from uncertainties import ufloat
from uncertainties import umath
from uncertainties.umath import *  # sin(), etc.

# %% [markdown]
# Define a quantity that hold the uncertainties value : 

# %%
height = ufloat(1.84, 0.1) 
qheight = height*m

print(height)
print(qheight)

# %% [markdown]
# Uncertainties attributes are still available but without unit (hence the need of a better interface):

# %%
print(qheight.nominal_value)
print(qheight.std_dev)
print(qheight.std_score(3))

# %% [markdown]
# Some operations fails like : 

# %%
u = ufloat(1, 0.1) #* m
v = ufloat(10, 0.1)# * m
sum_value = u + v
sum_value.derivatives[u]#]

# %% [markdown] tags=[]
# ## Operations with other quantities 
# are possible as long as uncertainties support the operation with the quantity's value : 

# %%
print(qheight*2)
print(qheight*2*m)
print(qheight**2)
print(qheight*np.arange(3))
print((2*qheight+1*m))

# %% [markdown]
# ## Operations with other uncertainties

# %% [markdown]
# By default, an uncertainty that is not wrapped by a quantity is supposed to have no physical dimension

# %%
print(qheight)
print(qheight * ufloat(2, 0.1))
print(qheight / ufloat(2, 0.1))


# %% [markdown]
# ## Access to the individual sources of uncertainty
# based on https://pythonhosted.org/uncertainties/user_guide.html#access-to-the-individual-sources-of-uncertainty
# Again, we loose the unit falling back on the backend value : we would like to have : 
# ```
# 21.00+/-0.22 m
# v variable: 0.2 m
# u variable: 0.1 m
# ```

# %%
u = ufloat(1, 0.1, "u variable") * m  # Tag
v = ufloat(10, 0.1, "v variable") * m
sum_value = u+2*v
print(sum_value)
for (var, error) in sum_value.error_components().items():
    print("{}: {}".format(var.tag, error))

# %% [markdown]
# # Comparison

# %%
x = ufloat(0.20, 0.01) *m
y = x + 0.0001*m

print(y > x) # expect True
print(y > 0*m) # expect True

y = ufloat(1, 0.1) * m
z = ufloat(1, 0.1) * m
print(y)
print(z)
print(y == y) # expect True
print(y == z) # expect False

# %% [markdown]
# # Math module and numpy
# Not tested but will most likely fails as uncertainties relies on `umath` and `unumpy`. To be fair, physipy also have a `math` module that wraps the builtin one.

# %% [markdown]
# # Dirty Sandbox

# %%
import uncertainties as uc
x =  uc.ufloat(0.20, 0.01)
arr = np.array([uc.ufloat(1, 0.01),
                uc.ufloat(2, 0.1)])

arr_q = np.array([arr[0]*m, arr[1]*m])
arr.sum() * m == arr_q.sum()

# %%
np.isclose(arr_q.sum().value, (arr.sum() * m).value)

# %%

# %%
arr_q.sum().value == (arr.sum() * m).value

# %% [markdown]
# # Measurement : mix between Uncertainties and Pint
# https://pint.readthedocs.io/en/stable/measurement.html?highlight=uncertainty

# %%
import numpy as np
book_length = (20. * m).plus_minus(2.)
print(book_length.value)
print(2 * book_length)


# %%
