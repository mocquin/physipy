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
# # Wrapped function
# A single class that wrap the function and a plot

# %%
# %matplotlib widget

from physipy import m, s
from physipy.qwidgets.plot_ui import WrappedFunction1D
from physipy.quantity.utils import name_eq

@name_eq("Myfunc")        
def func(x1, x2, x3):
    return x1*x2 + 3 * x3

wf = WrappedFunction1D(func, 0*s, 5*s, 
                       x2=(0*m, 5*m),
                       x3=(0*m*s, 5*m*s))

print(wf(1, 2, 3))

def add_integral():
    p = wf.add_integral(1*s, 5*s)
wf

# %%
from radiopy.radiation import planck_spec_en_
from physipy import units, K

mum = units["mum"]
wf = WrappedFunction1D(planck_spec_en_, 2*mum, 14*mum, 
                       T=(250*K, 340*K),
                       )

p = wf.add_integral(3*mum, 5*mum)

wf

# %%
