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
# # Numpy support for arrays with dimension

# %% [markdown]
# A Quantity object can have any numerical-like object as its `value` attribute, including numpy's ndarray.

# %% [markdown]
# Physipy support numpy for many functionnalties : 
#  - common creation routines
#  - mathematical operations
#  - numpy's functions and universal functions
#  - comparison
#  - indexing and fancy indexing
#  - iterators
#

# %% [markdown]
# ## Creation
# Basic creation of dimension-full arrays : 

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import physipy
from physipy import m, s, Quantity, Dimension, rad

# %%
x_samples = np.array([1, 2, 3, 4]) * m
y_samples = Quantity(np.array([1, 2, 3, 4]), Dimension("T"))
print(x_samples)
print(y_samples)
print(m*np.array([1, 2, 3, 4]) == x_samples) # multiplication is commutativ

# %% [markdown]
# ## Operation
# Basic array operation are handled the 'expected' way : note that the resulting dimension are consistent with the operation applied : 

# %%
print(x_samples + 1*m)
print(x_samples * 2)
print(x_samples**2)
print(1/x_samples)


# %% [markdown]
# ## Comparison

# %%
print(x_samples > 1.5*m)

try: 
    x_samples > 1.5*s
except Exception as e:
    print(e)


# %% [markdown]
# ## Numpy ufuncs
# Most numpy ufuncs are handled the expected way, but still check for dimension correctness :

# %%
q = 3*m
q_arr = np.arange(3)*m

print(np.add(q, q_arr))
print(np.multiply(q, q_arr))
print(np.sign(q_arr))
print(np.greater_equal(q_arr, 2*m))
print(np.sqrt(q_arr))
print(np.cbrt(q_arr))

print(np.cos(pi*rad))
print(np.tan(pi/4*rad))

print(np.ceil(q_arr**1.6))
print(np.negative(q_arr))

# %% [markdown]
# Trigonometric functions expect dimensionless quantities, and regular dimension correctness is expected : 

# %%
try:
    np.cos(3*m)
except Exception as e:
    print(e)

try:
    np.add(3*s, q_arr)
except Exception as e:
    print(e)

# %% [markdown]
# ## Numpy's functions

# %% [markdown]
# Most classic numpy's functions are also handled : 

# %%
print(np.linspace(3*m, 10*m, 5))
print(np.argmax(q_arr))
print(np.around(q_arr*2.3))
print(np.cross(q_arr, q_arr[::-1]))
print(np.dstack((q_arr, q_arr)))
print(np.mean(q_arr))
print(np.var(q_arr))
print(np.trapz(q_arr))
print(np.meshgrid(q_arr, q_arr))
print(np.fft.fft(q_arr))
print(np.convolve(q_arr, q_arr))
print(np.ravel(q_arr))

# %% [markdown]
# Reduce with ufuncs :

# %%
import numpy as np
from physipy import m
q = np.arange(10)*m

# %%
q = np.arange(10)*m
print(np.add.reduce(q))
print(np.multiply.reduce(q))

# %% [markdown]
# ## Indexing

# %%
big_arr = np.arange(20).reshape(4,5)*s

print(big_arr)
print(big_arr[0])
print(big_arr[:, 2])

# %% [markdown]
# ## Fancy indexing

# %%
print(big_arr)
print(np.greater_equal(big_arr, 12*s))
print(big_arr[np.greater_equal(big_arr, 12*s)])

# %% [markdown]
# ## Common array methods

# %% [markdown]
# ### flat iterator

# %%
print(big_arr.flat)

for q in q_arr.flat:
    print(q)


# %% [markdown]
# ## Known issues

# %% [markdown]
# ### logical fucntions

# %% [markdown]
# The expected behavior of logical functions is not trivial : 
#  - logical_and 
#  - logical_or
#  - logical_xor
#  - logical_not
#  
# Hence they are not implemented.

# %% [markdown]
# ### np.arange

# %% [markdown]
# The commonly used `np.arange` cannot be overriden the same way the ufuncs or classic numpy function can be. Hence, a wrapped version is provided

# %%
from physipy.quantity.utils import qarange

# %%
# using range
print(np.array(range(10))*m)
# using np.arange
print(np.arange(10)*m)
# using physipy's qarange : note that the "m" quantity is inside the function call
print(qarange(10*m))

# %% [markdown]
# With this wrapper, you can then do the following :

# %%
print(np.arange(2.5, 12)*m)
print(qarange(2.5*m, 12*m))

# %% [markdown]
# The qarange wrapper still cares about dimension correctness : 

# %%
try:
    print(qarange(2*m, 10*s))
except Exception as e:
    print(e)

# %%
np.reshape(q_arr, (1, len(q_arr)))
