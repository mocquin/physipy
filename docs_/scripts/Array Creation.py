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
import numpy as np
import physipy
from physipy import m, units
from physipy.quantity.utils import asqarray

# %% [markdown]
# # Array-like creation

# %% [markdown]
# Several creation will fail : 
#  - non homomgeneous (obviously)
#  - but also in-array creation

# %%
length = 3*m
#arr = np.array([
#    [0*m, length]
#])

# %%
arr = asqarray([0*m, length])
arr

# %%
#np.asarray([[0*m, 3*m]], like=m)

# %%
arr = asqarray([[0*m, length],[0*m, length]])
arr

# %%
list([[0*m, 2*m], [2*m, 2*m]])

# %%
np.array(0*m)

# %%
# %timeit asqarray([0*m, 2*m])
# %timeit [0, 2]*m

# %%
a = [[0*m, 2*m]]

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
