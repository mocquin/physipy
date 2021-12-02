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

# %%
import numpy as np
import physipy
from physipy import m, units
from physipy import array_to_Q_array
from physipy.quantity.utils import list_of_Q_to_Q_array, asqarray

# %% [markdown]
# # Array-like creation

# %% [markdown]
# Several creation will fail : 
#  - non homomgeneous (obviously)
#  - but also in-array creation

# %%
length = 3*m
arr = np.array([
    [0*m, length]
])

# %%
arr = asqarray([0*m, length])
arr

# %%
np.asarray([[0*m, 3*m]], like=m)

# %%
arr = asqarray([[0*m, length],[0*m, length]])
arr

# %%
list([[0*m, 2*m], [2*m, 2*m]])

# %%
np.array(0*m)

# %%
from physipy import Quantity

# %%
np.array([2.4*m, 3.4*m])

# %%
# %timeit asqarray([0*m, 2*m])
# %timeit [0, 2]*m

# %%
a = [[0*m, 2*m]]

# %%

    
    
        

# %%
a = [[0*m, 2*m], [0*m, 2*m]]
b = [[0*m, 2*m]]
c = [0*m]
d = [1*m, 2*m]
print(_wrap(a))
print(_wrap(b))
print(_wrap(c))
print(_wrap(d))


# %%
def my_flat(a):
    if isinstance(a, list):
        return [e if not isinstance(a, list) else a for e in a ]
    else:
        a

print(my_flat(a))
print(my_flat(b))
print(my_flat(c))
print(my_flat(d))



# %%
def flat(a):
    return reduce(lambda x,y: x+y, newlist)


# %%

# %%
import collections




# %%
flatten([[1, 2, 3], [2, 3]])

# %%
flatten(a)

# %%
x = m

# %%
# %timeit np.array([[0, -x.value/2], [1, x.value]])*m
# %timeit asqarray([[0*m, -x/2], [1*m, x],])

# %%

# %%
shape([[[0*m, 2*m],
      [2*m, 3*m]]])

# %%

# %%
from collections.abc import Sequence, Iterator
from itertools import tee, chain

def is_shape_consistent(lst: Iterator):
    """
    check if all the elements of a nested list have the same
    shape.

    first check the 'top level' of the given lst, then flatten
    it by one level and recursively check that.

    :param lst:
    :return:
    """

    lst0, lst1 = tee(lst, 2)

    try:
        item0 = next(lst0)
    except StopIteration:
        return True
    is_seq = isinstance(item0, Sequence)

    if not all(is_seq == isinstance(item, Sequence) for item in lst0):
        return False

    if not is_seq:
        return True

    return is_shape_consistent(chain(*lst1))
