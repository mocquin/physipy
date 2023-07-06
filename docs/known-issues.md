---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Known issues


## A float quantity is Iterable
https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
This tests if the object has "__iter__"

<!-- #region -->
 - iter(x): on quantity doesn't raise a TypeError just because Quantity has a method called '__getitem__'. Would also if __iter__ was a method
 - np.iterable : (used by matplotlib units interface):
```python 
try:
    iter(y)
except TypeError:
    return False
return True
```
 -  https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
 
```
class collections.abc.Iterable¶

    ABC for classes that provide the __iter__() method.

    Checking isinstance(obj, Iterable) detects classes that are registered as Iterable or that have an __iter__() method, but it does not detect classes that iterate with the __getitem__() method. The only reliable way to determine whether an object is iterable is to call iter(obj).

```
<!-- #endregion -->

```python
import collections
from physipy import m

isinstance(m, collections.abc.Iterable)

import numpy as np
np.iterable(m)
```

As a comparison for the underlying value : 

```python
print(type(m.value))
print(iter(m))
```

## Array repr with 0 value
Pick best favunit take the smallest when 0 is in the array with positive and negative values :

```python
from physipy import m, Quantity, Dimension
import numpy as np
Quantity(np.array([0, -1.2, 1.2]), Dimension("L"))
```

# Inplace change using asqarray

```python
from physipy.quantity.utils import asqarray
print(type(m.value))
arrq_9 = np.array([m.__copy__()], dtype=object)
out = asqarray(arrq_9)
# this changes the type of m value
print(type(m.value))
```

# Numpy trapz implementaion not called when only x or dx is a quantity


Only work when the array to integrate is a quantity. An issue is open at : https://github.com/numpy/numpy/issues/18902.

```python
from physipy import m
import numpy as np
```

```python
# this works
print(np.trapz(np.arange(5)*m))
# also this
print(np.trapz(np.arange(5), x=np.arange(5)*m))
print(np.trapz(np.arange(5), dx=5000*m, x=np.arange(5)*m)) #dx is silent
# but not this
# np.trapz(np.arange(5), dx=5000*m)
print("----uncomment above line to trigger exception")
```

# Array function interface not triggered on scalars


Calling a numpy function with only scalars will not trigger the array function interface, since it is used only when an argument is an array.

```python
from physipy import m
# this raises a DimensionError because of the casting into float
#np.random.normal(3*m, 1*m)
# while this works
np.random.normal(np.array(3*m), np.array(1*m))
```

# HALF-FIXED (Matplotlib histogram) by adding "to_numpy" method, but we loose the unit


It turns out that matplotlib first checks if the object has a "to_numpy()" method, then again improved by removing to_numpy and removing __iter__ and delegate it to getattr


Some preprocessing turn a quantity-array into a "set of elements", and plots one histogram for each value.

```python
import numpy as np
from physipy import m
import matplotlib.pyplot as plt
```

```python
arr = np.random.normal(1, 0.1, size=100)*m
arr
```

```python
plt.hist(arr.value)
plt.hist(arr)
```

```python
plt.hist(np.arange(10)*m)
```

# Matplotlib histogram, again : missing units support

<!-- #region -->
Source code for hist : https://matplotlib.org/stable/_modules/matplotlib/axes/_axes.html#Axes.hist

One of the first things done is : 
```python
x = cbook._reshape_2D(x, 'x')
```
With 
```python
Signature: cbook._reshape_2D(X, name)
Source:   
def _reshape_2D(X, name):
    """
    Use Fortran ordering to convert ndarrays and lists of iterables to lists of
    1D arrays.

    Lists of iterables are converted by applying `np.asanyarray` to each of
    their elements.  1D ndarrays are returned in a singleton list containing
    them.  2D ndarrays are converted to the list of their *columns*.

    *name* is used to generate the error message for invalid inputs.
    """

    # unpack if we have a values or to_numpy method.
    try:
        X = X.to_numpy()
    except AttributeError:
        try:
            if isinstance(X.values, np.ndarray):
                X = X.values
        except AttributeError:
            pass

    # Iterate over columns for ndarrays.
    if isinstance(X, np.ndarray):
        X = X.T

        if len(X) == 0:
            return [[]]
        elif X.ndim == 1 and np.ndim(X[0]) == 0:
            # 1D array of scalars: directly return it.
            return [X]
        elif X.ndim in [1, 2]:
            # 2D array, or 1D array of iterables: flatten them first.
            return [np.reshape(x, -1) for x in X]
        else:
            raise ValueError(f'{name} must have 2 or fewer dimensions')

    # Iterate over list of iterables.
    if len(X) == 0:
        return [[]]

    result = []
    is_1d = True
    for xi in X:
        # check if this is iterable, except for strings which we
        # treat as singletons.
        if (isinstance(xi, collections.abc.Iterable) and
                not isinstance(xi, str)):
            is_1d = False
        xi = np.asanyarray(xi)
        nd = np.ndim(xi)
        if nd > 1:
            raise ValueError(f'{name} must have 2 or fewer dimensions')
        result.append(xi.reshape(-1))

    if is_1d:
        # 1D array of scalars: directly return it.
        return [np.reshape(result, -1)]
    else:
        # 2D array, or 1D array of iterables: use flattened version.
        return result
```
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
from physipy import units, m, K, setup_matplotlib
from matplotlib import cbook
import numpy as np
```

```python
arr = np.random.normal(1, 0.1, size=100)*m
setup_matplotlib()
```

```python
plt.hist(arr)
```

I created a [unit package for physical computation](https://github.com/mocquin/physipy) and its [matplotlib' unit interface](https://github.com/mocquin/physipy/blob/master/physipy/quantity/plot.py) that works well for plotting with `Axes` methods like `ax.plot`, as you can see in the [plotting notebooke demo](https://github.com/mocquin/physipy/blob/master/docs/Plotting.ipynb).
The issue I am facing is to have the unit interface work with the histogram plotting, like in `ax.hist(arr)`.

For now, I have 2 solutions that are not satisfiying : 
 - first solution is the current state of my project : I added to my `Quantity` object a `to_numpy()` method that cast the instance into a plain numpy array, which makes the histogram plotting work, but looses the automatic unit plotting, since it is not a Quantity anymore but a plain numpy array. For some reasons, this method is never used when plotting with `ax.plot`, but is one of the first things tried when using `ax.hist`.
 - other solution is what I had until recently, which was even worse : without the `to_numpy()` method, matplotlib tries to loop inside the object, and since my object can be iterated over (if it is a 1D array for eg), then it plots one 1-element-histogram for each value. You can see what it looked like [here](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=f0871009f57da092eee1d640d9508070d1662c1d&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d6f637175696e2f706879736970792f663038373130303966353764613039326565653164363430643935303830373064313636326331642f646f63732f4b6e6f776e2532304973737565732e6970796e62&nwo=mocquin%2Fphysipy&path=docs%2FKnown+Issues.ipynb&repository_id=175999533&repository_type=Repository#Matplotlib-histogram) (see the Matplotlib histogram section).

After digging into the source code, I found that some preprocessing is done the object passed to `hist`, using [cbook._reshape2d](https://github.com/matplotlib/matplotlib/blob/6f92db0fc6aad8dfcdb197202b969a01e4391fde/lib/matplotlib/axes/_axes.py#L6654). Then, onto [`cbook._reshape2D` source](https://github.com/matplotlib/matplotlib/blob/6f92db0fc6aad8dfcdb197202b969a01e4391fde/lib/matplotlib/cbook/__init__.py#L1304), I think most of the time objects are subclass of `np.ndarray`, and so are caught [in this loop](https://github.com/matplotlib/matplotlib/blob/6f92db0fc6aad8dfcdb197202b969a01e4391fde/lib/matplotlib/cbook/__init__.py#L1327). But my class is not a subclass of `np.ndarray`, so it ends up [in this loop](https://github.com/matplotlib/matplotlib/blob/6f92db0fc6aad8dfcdb197202b969a01e4391fde/lib/matplotlib/cbook/__init__.py#L1347). Then `np.asanyarray(xi)` is called and cast each quantity element (like `1m` into just `1`) into an int/float, again loosing the unit. Now the workaround for this kind of problem is currently addressed by numpy's [NEP-35](https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html), which allows to override array-creation functions (like `np.asanyarray`), but it requires to pass an extra argument : `np.asanyarray(x)` would be `np.asanyarray(x, like=x)`. Unfortunately, this doesn't solve completely the problem, because the `xi` object are reshaped using `xi.reshape` and not `np.reshape(xi)`.

```python
import collections
import numpy as np
from physipy import m

# make a 1D array of meters
X = np.arange(10)*m

# introspect https://github.com/matplotlib/matplotlib/blob/6f92db0fc6aad8dfcdb197202b969a01e4391fde/lib/matplotlib/cbook/__init__.py#L1347 
result = []
is_1d = True
for xi in X:
    # check if this is iterable, except for strings which we
    # treat as singletons.
    if (isinstance(xi, collections.abc.Iterable) and
             not isinstance(xi, str)):
        is_1d = False
    #xi = np.asanyarray(xi)
    print(xi)
    xi = np.asanyarray(xi, like=xi)
    nd = np.ndim(xi)
    print(xi.reshape(-1))
    if nd > 1:
        raise ValueError(f'{name} must have 2 or fewer dimensions')
    result.append(xi.reshape(-1))

print(is_1d)
if is_1d:
    # 1D array of scalars: directly return it.
    #return 
    print([np.reshape(result, -1)])
else:
    # 2D array, or 1D array of iterables: use flattened version.
    #return 
    print(result)
```

# Numpy random normal


Not really a bug, more of a Feature request on numpy's side...
`__array_struct__` is tried on the value, and unit is dropped.

See https://github.com/numpy/numpy/issues/19382

```python
import numpy as np
from physipy import m
```

```python
np.random.normal(np.array([1, 2, 3]),
                 np.array([2, 3, 4]), size=(2, 3))
```

```python
np.random.normal(np.array([1, 2, 3])*m,
                 np.array([2, 3, 4])*m, size=(2, 3))
```

```python
import numpy as np
np.random.seed(1234)

HANDLED_FUNCTIONS = {}

class NumericalLabeled():
    def __init__(self, value, label=""):
        self.value = value
        self.label = label
        
    def __repr__(self):
        return "NumericalLabelled<"+str(self.value) + "," + self.label+">"
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    

def make_numericallabelled(x, label=""):
    """
    Helper function to cast anything into a NumericalLabelled object.
    """
    if isinstance(x, NumericalLabeled):
        return x
    else:
        return NumericalLabeled(x, label=label)
    
# Numpy functions            
# Override functions - used with __array_function__
def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator    
    

@implements(np.random.normal)
def np_random_normal(loc=0.0, scale=1.0, **kwargs):
    # cast both loc and scale into Numericallabelled
    loc = make_numericallabelled(loc)
    scale = make_numericallabelled(scale)
    # check their label is "compatible"
    if not loc.label == scale.label:
        raise ValueError
    return NumericalLabeled(np.random.rand(loc=loc.value,
                                           scale=scale.value, **kwargs), 
                            loc.label+scale.label)

@implements(np.mean)
def np_mean(a, *args, **kwargs):
    return NumericalLabeled(np.mean(a.value, *args, **kwargs),
                            a.label)



def main():
    # reference result for standard array
    arr = np.arange(10)
    print(np.mean(arr))
    print(np.random.normal(arr))
    
    # array-like object
    num_labeled = NumericalLabeled(arr, "toto")
    print(np.mean(num_labeled))
    try:
        print(np.random.normal(num_labeled))
    except Exception as e:
        print(e)

main()
```

```python
import sys
print("Python :", sys.version)
print("Numpy :", np.__version__)
```

```python
arr = np.arange(10)

# Reference result 
print(np.mean(arr))
print(np.random.normal(arr))

custom_obj = MyArrayLike(arr)
print(np.mean(custom_obj))           # np.mean will trigger __array_function__ interface
print(np.random.normal(custom_obj))  # np.random.normal will "only" try to cast the object to float
```

# Power of dimension are not casted to int when possible

```python
from physipy import m, K
```

```python
a = (m**2)**0.5
a.dimension
```

# Degree rendering


Degree as favunit is rendered as "<function deg at 0x...>"

```python
from physipy import rad, units
deg = units["deg"]
a = 5*deg
a.favunit = deg
a
```

# Numpy full not triggered by array_function interface


Get `TypeError: no implementation found for 'numpy.full' on types that implement __array_function__: [<class 'physipy.quantity.quantity.Quantity'>]`
while it is implemented, so it seems its not triggered.

This is a numpy bug : https://github.com/numpy/numpy/issues/21033

```python
from physipy import m
import numpy as np

np.full(3, m, like=m)
```

# Presence of "%" symbol in favunit

```python
from physipy import K
import sympy as sp
pc = 1/K
pc.symbol = "%/K"

q = 2*pc
q.favunit = pc
```

```python
pc.symbol
```

```python
complemented = q._compute_complement_value().encode('unicode-escape').decode()
complemented
```

```python
a = sp.Symbol("a")
b = sp.physics.units.percent #sp.Symbol("%")
```

```python
a/b
```

```python
((percent_transformer,) + standard_transformations )
```

```python
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations 

def percent_transformer(tokens, local_dict, global_dict):
    return [tok if tok != "%" else  sp.physics.units.percent for tok in tokens]

sp.parsing.sympy_parser.parse_expr("%/K", transformations=((percent_transformer,) + standard_transformations))
```

```python
sp.parsing.sympy_parser.parse_expr(complemented,
                                   local_dict={"%":sp.physics.units.percent},
                                   #evaluate=False, 
                              )
```


https://github.com/numpy/numpy/issues/18902


Numpy trapz bug

```python
import numpy as np

HANDLED_FUNCTIONS = {}

class NumericalLabeled():
    def __init__(self, value, label=""):
        self.value = value
        self.label = label
        
    def __repr__(self):
        return "NumericalLabelled<"+str(self.value) + "," + self.label+">"
    
    def __array_function__(self, func, types, args, kwargs):
        print("Got into array function")
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
def make_numericallabelled(x, label=""):
    """
    Helper function to cast anything into a NumericalLabelled object.
    """
    if isinstance(x, NumericalLabeled):
        return x
    else:
        return NumericalLabeled(x, label=label)
    
# Numpy functions            
# Override functions - used with __array_function__
def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator    
    
@implements(np.trapz)
def np_trapz(q, x=None, dx=1, **kwargs):
    """
    Numpy's trapz wrapper for NumericalLabelled.
    """
    # first convert q into a NumericalLabelled to use `q.value` 
    q = make_numericallabelled(q)
    if x is None:    
        # using dx.value and dx.label
        dx = make_numericallabelled(dx, label="dx")
        return NumericalLabeled(np.trapz(q.value, dx=dx.value, x=None, **kwargs),
                                q.label + dx.label,
                    )
    else:
        # using x/value and x.label
        x = make_numericallabelled(x, label="x")
        return NumericalLabeled(np.trapz(q.value, x=x.value, **kwargs),
                                q.label + x.label,
                    )

def main():
    # create a scalar to use as dx
    half = NumericalLabeled(0.5, "half")
    # create an array to use as x
    x = NumericalLabeled(np.arange(5), "x")
    # then 
    # this works
    print(np.trapz(NumericalLabeled(np.arange(5), "a")))
    # this also works
    print(np.trapz(np.arange(5), x=x))
    # but not this
    print(np.trapz(np.arange(5), dx=half))
    # TypeError: unsupported operand type(s) for *: 'NumericalLabeled' and 'int'
main()
```

```python
import sys, numpy; print(numpy.__version__, sys.version)
```

# random.normal array_function

```python
import numpy as np

HANDLED_FUNCTIONS = {}

class NumericalLabeled():
    def __init__(self, value, label=""):
        self.value = value
        self.label = label
        
    def __repr__(self):
        return "NumericalLabelled<"+str(self.value) + "," + self.label+">"
    
    def __array_function__(self, func, types, args, kwargs):
        print("Got into array function")
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
def make_numericallabelled(x, label=""):
    """
    Helper function to cast anything into a NumericalLabelled object.
    """
    if isinstance(x, NumericalLabeled):
        return x
    else:
        return NumericalLabeled(x, label=label)
    
# Numpy functions            
# Override functions - used with __array_function__
def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator    
    

@implements(np.random.normal)
def np_random_normal(loc=0.0, scale=1.0, **kwargs):
    # cast both loc and scale into Numericallabelled
    loc = make_numericallabelled(loc)
    scale = make_numericallabelled(scale)
    # check their label is "compatible"
    if not loc.label == scale.label:
        raise ValueError
    return NumericalLabeled(np.random.rand(loc=loc.value,
                                           scale=scale.value, **kwargs), 
                            loc.label+scale.label)



def main():
    # create two scalars
    half = NumericalLabeled(0.5, "half")
    loc = np.array(half)
    print(np.random.normal(loc=loc))
    # this raises a TypeError : 
    print(np.random.normal(loc=half))
    # TypeError: unsupported operand type(s) for *: 'NumericalLabeled' and 'int'

main()
```
