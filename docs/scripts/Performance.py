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

# %%
# %%timeit 
try:
    pass
except:
    pass

# %%
# %%timeit
isinstance(m, Quantity)

# %% [markdown] tags=[]
# ## benchmark timing

# %% [markdown]
# Here is a comparison of most operations : 

# %%
import numpy as np
from physipy import m, s, Quantity
arr = np.linspace(0, 200)
sca = 5.14
pi = np.pi
arr_m = arr * m


ech_lmbda_mum = np.linspace(2, 15)

def bench_scalar_op_add(): m + m
def bench_scalar_op_sub(): m - m
def bench_scalar_op_mul(): m * m
def bench_scalar_op_div(): m / m
def bench_scalar_op_truediv(): m // m
def bench_scalar_op_pow(): m ** 1
def use_case():
    x = arr * m
    x2 = sca * s**2
    y = x*x2/pi * np.sum(x**2) + 3*m**3*s**2

def bench_arr_scalar_op_add(): arr_m + m
def bench_arr_scalar_op_sub(): arr_m - m
def bench_arr_scalar_op_mul(): arr_m * m
def bench_arr_scalar_op_div(): arr_m / m
def bench_arr_scalar_op_truediv(): arr_m // m
def bench_arr_scalar_op_pow(): arr_m ** 1
def use_case2():
    from physipy import units, constants, K
    mum = units["mum"]
    hp = constants["h"]
    c = constants["c"]
    kB = constants["k"]
    
    def plancks_law(lmbda, Tbb):
        return 2*hp*c**2/lmbda**5# * 1/(np.exp(hp*c/(lmbda*kB*Tbb))-1)
    lmbdas = ech_lmbda_mum*mum
    Tbb = 300*K
    integral = np.trapz(plancks_law(lmbdas, Tbb), x=lmbdas)


# %%
import pint
import physipy
import forallpeople
import numpy as np
from astropy import units as astropy_units

ureg = pint.UnitRegistry()

a = 123
b = 654

# we don't want a big array since we are intersted in the differences
# between scalars and arrays, if any. Big array length could introduce
# time deltas from the numerical computation, not the unit overhead
arr = np.arange(1, 4)

import timeit

import operator

# %%
operations = {
    "add":"__add__", 
    "sub":"__sub__",
    "mul":"__mul__",
    "truediv":"__truediv__",
}
N = 100000

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
asp_qs = {
    "name":"astropy",
    "a":a*astropy_units.m,
    "b":b*astropy_units.m,
    "arrm":arr*astropy_units.m,
}

# %%
results = []

for modules_dict in [physipy_qs, pint_qs, fap_qs, asp_qs]:
    print(modules_dict["name"])
    for operation, operation_method in operations.items():
        aq = modules_dict["a"]
        bq = modules_dict["b"]
        #time = timeit.timeit('a.'+operation_method+"(b)", number=10000, globals=globals())
        time_q = timeit.timeit('aq.'+operation_method+"(bq)", number=N, globals={"aq":aq, "bq":bq})#), globals=globals())        
        #print(f"{operation: >5} : {time_q/time: <5.1f}")
        print(f"{operation :>5} : {time_q:.5f}")
        results.append((modules_dict['name'], operation, time_q, aq, bq, "sca", "sca", str(getattr(operator, operation)(aq, bq))))
    for operation, operation_method in operations.items():
        aq = modules_dict["a"]
        arrq = modules_dict["arrm"]
        #time = timeit.timeit('a.'+operation_method+"(b)", number=10000, globals=globals())
        try:
            time_qarr = timeit.timeit('aq.'+operation_method+"(arr)", number=N, globals={"aq":aq, "arr":arrq})#), globals=globals())      
            print(f"{operation :>5} : {time_qarr:.5f}")
            results.append((modules_dict['name'], operation, time_q, aq, arrq, "sca", "arr", str(getattr(operator, operation)(aq, arrq))))

        except Exception as e:
            print(f"{operation :>5} : {e}")
            
            results.append((modules_dict['name'], operation, np.nan, aq, arrq, "sca", "arr", "Failed"))
import pandas as pd
import seaborn as sns
df = pd.DataFrame(results, columns=["package", "op", "time", "left", "right", "left_type", "right_type", "result"])
sns.catplot(
    col="op", x="right_type", y="time", hue="package", data=df, kind="bar")

# %%
physipy
  add : 0.26833
  sub : 0.24462
  mul : 0.96287
truediv : 1.05938
  add : 0.33669
  sub : 0.33431
  mul : 1.21743
truediv : 1.50563

# %%
aq=a*physipy.m
arrm = arr*physipy.m

# %%
# %timeit arr*a
# %timeit aq*arrm

# %%
arrm

# %%
physipy
  add : 0.57792
  sub : 0.58624
  mul : 1.10884
truediv : 1.20130
  add : 0.69919
  sub : 0.74585
  mul : 1.34999
truediv : 1.72294

physipy
  add : 0.58849
  sub : 0.59637
  mul : 1.10718
truediv : 1.19365
  add : 0.75698
  sub : 0.79829
  mul : 1.34892
truediv : 1.63579

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


# %% [markdown]
# ## Array creation

# %% [markdown]
# Compare lazy creation of arrays

# %%
# %timeit asqarray([0*m, 2*m])
# %timeit [0, 2]*m

# %% [markdown]
# ## Profiling base operations

# %%
from physipy import m, s, rad, sr

# %%
bench_scalar_op_add
bench_scalar_op_mul
bench_arr_scalar_op_add
bench_arr_scalar_op_mul

# %%
# #%prun -D prunsum -s time m+m 
# %prun -D prunsum_file -s nfl use_case()
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
# ## Profiling tests

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
