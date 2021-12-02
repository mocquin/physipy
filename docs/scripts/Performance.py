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
# # Performance

# %% [markdown]
# Obviously, using unit-aware variables will slow down any computation compared to raw python values (int, flot, numpy.ndarray).

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
mus = units["mus"]
ns = units["ns"]
a = 123456
b = 654321

aq = a*m
bq = b*m


# %% [markdown]
# Basic comparison on addition

# %%
# %timeit  (a +  b)
# %timeit (aq + bq)

# %%
print(12.4*mus/(63.7*ns))

# %% [markdown]
# Basic comparison on pow

# %%
# %timeit  (a**2)
# %timeit (aq**2)

# %%
print(22.8*mus/(289*ns))

# %% [markdown]
# ## benchmark timing

# %% [markdown]
# Here is a comparison of most operations : 

# %%
import timeit

operations = {
    "add":"__add__", 
    "sub":"__sub__",
    "mul":"__mul__",
}

# %%
import pint
import physipy
import forallpeople
import numpy as np

ureg = pint.UnitRegistry()

a = 123456
b = 654321
arr = np.arange(100)

# %%
physipy_qs = {
    "name":"physipy",
    "a":a*physipy.m,
    "b":b*physipy.m,
    'arrm':arr*physipy.m,
}
pint_qs = {
    "name":"pint",
    "a":a*ureg.m,
    "b":b*ureg.m,
    'arrm':arr*ureg.m,
}
fap_qs = {
    "name":"forallpeople",
    "a":a*forallpeople.m,
    "b":b*forallpeople.m, 
    'arrm':arr*forallpeople.m, 
}

for modules_dict in [physipy_qs, pint_qs, fap_qs]:
    print(modules_dict["name"])
    for operation, operation_method in operations.items():
        aq = modules_dict["a"]
        bq = modules_dict["b"]
        #time = timeit.timeit('a.'+operation_method+"(b)", number=10000, globals=globals())
        time_q = timeit.timeit('aq.'+operation_method+"(bq)", number=10000, globals=globals())        
        #print(f"{operation: >5} : {time_q/time: <5.1f}")
        print(f"{operation :>5} : {time_q:.5f}")
    for operation, operation_method in operations.items():
        aq = modules_dict["a"]
        arr = modules_dict["arrm"]
        #time = timeit.timeit('a.'+operation_method+"(b)", number=10000, globals=globals())
        time_qarr = timeit.timeit('aq.'+operation_method+"(arr)", number=10000, globals=globals())
        
        #print(f"{operation: >5} : {time_q/time: <5.1f}")
        print(f"{operation :>5} : {time_qarr:.5f}")
    

# %%



# %%

# %%

# %% [markdown]
#

# %%
pip install line_profiler

# %%
# %lprun

# %% [markdown]
# # Array creation

# %% [markdown]
# Compare lazy creation of arrays

# %%
# %timeit asqarray([0*m, 2*m])
# %timeit [0, 2]*m

# %% [markdown]
# # Profiling base operations

# %%
from physipy import m, s, rad, sr

# %%
# #%prun -D prunsum -s time m+m 
# %prun -D prunsum_file -s nfl m+m 
# !snakeviz prunsum_file

# %% [markdown]
# Ideas for better performances : 
#  - less 'isinstance'
#  - remove sympy 
#  - cleaner 'setattr'

# %%
# %%prun -s cumulative -D prundump
m + m
2 * m
2*s /(3*m)
m**3

# %%
# !snakeviz prundump

# %% [markdown]
# # Profiling tests

# %%
import sys
sys.path.insert(0,r"/Users/mocquin/MYLIB10/MODULES/physipy/test")
import physipy
import test_dimension
import unittest
from physipy import Quantity, Dimension

# %%
from test_dimension import TestClassDimension
from test_quantity import TestQuantity

# %%
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestQuantity)

# %%
# %%prun -s calls -D prun
unittest.TextTestRunner().run(suite)

# %%
# !snakeviz prun

# %%
# %timeit Quantity(1, Dimension(None))

# %%
# %timeit Quantity(1, Dimension(None))

# %%
testdim = test_dimension.TestClassDimension()

# %%
testdim.run()

# %%
# %cd ..

# %%
# #%cd
# %prun -s module !python -m unittest

# %%

# %%
