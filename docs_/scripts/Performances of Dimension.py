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
# # Dimension performance

# %%
import sys
sys.path.append(r"/Users/mocquin/MYLIB10/MODULES/simparser/")

# %%
from physipy import Dimension
from physipy.quantity.dimension import parse_str_to_dic
from simparser import new_parse_str_to_dict

# %%
# %timeit parse_str_to_dic("M*L**2/T**3*I**-1")
# %timeit new_parse_str_to_dict("M*L**2/T**3*I**-1")

# %%
# %timeit Dimension({"M":1, "L":2, "T":-3, "I":-1})
# %timeit Dimension("M*L**2/T**3*I**-1")

# %%
# %prun -D prunsum_file -s nfl  new_parse_str_to_dict("M*L**2/T**3*I**-1")
# !snakeviz prunsum_file

# %%
from physipy import Dimension

# %%
d = Dimension("L")

# %%
# %prun -D prun d*d

# %%
# !snakeviz prun

# %% [markdown]
# We need operations on array-like objects.
# The solutions are :
#  - a dict
#  - list
#  - numpy array
#  - ordered dict
#  - counter
# Among these solutions

# %% [markdown]
# Most important operators : 
#  - equality check, to check if the dimensions are equal (for `Dimension.__eq__`)
#  - addition of values key-wise, when computing the product of 2 dimension (for `Dimension.__mul__`)
#  - substration of values key-wise, when computing the division of 2 dimensions (for `Dimension.__truediv__`)
#  - multiplication of all values, when computing the exp of a dimension by a scalar (for `Dimension.__pow__`)
#  

# %% [markdown]
# We can rely on the operators, but the actual implementation matters. Exemple for 

# %%
import operator as op
operators = {
     "op.eq":("binary", op.eq), 
    "op.add":("binary", op.add),
    "op.sub":("binary", op.sub),
    "op.mul":("binary", op.mul),
}

# %%
import time
class Timer():
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs


# %%
class Implem():
    def __init__(self, name, creator):
        self.name = name
        self.creator = creator
    def __call__(self, *args, **kwargs):
        return self.creator(*args, **kwargs)

    
    
implemetations = [DimAsDict, DimAsArray, DimAsList]
    
def bench_dimension_base_data(ns=[3, 4, 5, 6, 7, 8, 10, 15, 20, 50, 100, 1000, 10000]):
    # 4 operations to time
    # for various number of dimensions 
    # for all implemetations
    # need to store the result of each test
    res = []
    for implem in implemetations:
        for opmeta in operators:            
            for n in ns:
                obj = implem(n)
                if opmeta[0] == "binary":
                    op = opmeta[1]
                    with Timer() as t:
                        resop = op(obj, obj)
                res_dict = {
                    "implem":implem.name,
                    "n":n,
                    "result":resop,
                    "time":t.msecs,
                }
                res.append(res_dict)
                    
                    
                

# %%
import physipy
from physipy import m, Dimension

# %%
d = Dimension("L")
# %timeit d**2

# %%
d = Dimension("L")
# %timeit d**2

# %%


# %%
import numpy as np

class DimAsListArray():
    """
    Benefit the speed of array when computing mul/div, and speed of list equality for keys
    """
    
    def __init__(self, values=np.zeros(3), KEYS=BASEKEYS):
        self.dims_keys = KEYS
        self.dim_values = values
        
    def __mul__(self, other):
        return DimAsListArray(self.dim_values+other.dim_values)
        


# %%

# %%
import numpy as np
import collections
"""Goal : return True if 2 vectors of numbers are equal
Inputs :
 - vectors are assured to be the same size
 - vector values can be int, float, np.numbers, fractions
 - the order of the numbers matters (like with dict comparison or ordered dict)
"""
 
as_dictl = {"A":0, "B":0, "C":0}
as_dictr = {"A":0, "B":0, "C":0}
as_listl = [0, 0, 0]
as_listr = [0, 0, 0]
as_arryl = np.array([0, 0, 0])
as_arryr = np.array([0, 0, 0])
as_odictl = collections.OrderedDict( {"A":0, "B":0, "C":0})
as_odictr = collections.OrderedDict( {"A":0, "B":0, "C":0})
as_counterl = collections.Counter("AAABBBCCC")
as_counterr = collections.Counter("AAABBBCCC")

# %%
# %timeit as_listl == as_listr
# %timeit as_dictl == as_dictr
# %timeit as_counterl == as_counterr
# %timeit as_odictl == as_odictr
# %timeit as_arryl.tolist() == as_arryr.tolist()
# %timeit list(as_odictl.values()) == list(as_odictr.values())
# %timeit np.array_equal(as_arryl, as_arryr)
# %timeit np.all(as_arryl == as_arryr)


# %%
a = np.arange(500)
b = np.arange(500)

# %timeit np.all(a == b)
# %timeit a.tolist() == b.tolist()


# %%
import numpy as np
import collections
from operator import add


as_dictl = {"A":0, "B":0, "C":0}
as_dictr = {"A":0, "B":0, "C":0}
as_listl = [0, 0, 0]
as_listr = [0, 0, 0]
as_arryl = np.array([0, 0, 0])
as_arryr = np.array([0, 0, 0])
as_odictl = collections.OrderedDict( {"A":0, "B":0, "C":0})
as_odictr = collections.OrderedDict( {"A":0, "B":0, "C":0})

# %timeit [l+r for l,r in zip(as_listl, as_listr)]
# %timeit {k:as_dictl[k]+as_dictr[k] for k in (as_dictl.keys() & as_dictr.keys())}
# #%timeit as_odictl == as_odictr
# #%timeit as_arryl.tolist() == as_arryr.tolist()
# #%timeit list(as_odictl.values()) == list(as_odictr.values())
# #%timeit np.array_equal(as_arryl, as_arryr)
# %timeit as_arryl + as_arryr
# %timeit list(map(add, as_listl, as_listr))

# %%

# %%

# %%
import numpy as np
import collections
from operator import mul

as_dictl = {"A":0, "B":0, "C":0}
as_dictr = {"A":0, "B":0, "C":0}
as_listl = [0, 0, 0]
as_listr = [0, 0, 0]
as_arryl = np.array([0, 0, 0])
as_arryr = np.array([0, 0, 0])
as_odictl = collections.OrderedDict( {"A":0, "B":0, "C":0})
as_odictr = collections.OrderedDict( {"A":0, "B":0, "C":0})

# %timeit [l*r for l,r in zip(as_listl, as_listr)]
# %timeit {k:as_dictl[k]*as_dictr[k] for k in (as_dictl.keys() & as_dictr.keys())}
# #%timeit as_odictl == as_odictr
# #%timeit as_arryl.tolist() == as_arryr.tolist()
# #%timeit list(as_odictl.values()) == list(as_odictr.values())
# #%timeit np.array_equal(as_arryl, as_arryr)
# %timeit as_arryl * as_arryr
# %timeit list(map(mul, as_listl, as_listr))

# %%
import numpy as np
import collections
from operator import pow

as_dictl = {"A":1, "B":1, "C":1}
as_dictr = 2
as_listl = [1, 1, 1]
as_listr = 2
as_arryl = np.array([1, 1, 1])
as_arryr = 2
as_odictl = collections.OrderedDict( {"A":1, "B":1, "C":1})
as_odictr = 2

# %timeit [l**as_dictr for l in as_listl]
# %timeit {k:as_dictl[k]**as_dictr for k in as_dictl.keys()}
# %timeit as_arryl ** as_arryr
# %timeit list(map(lambda x:x**2, as_listl))
