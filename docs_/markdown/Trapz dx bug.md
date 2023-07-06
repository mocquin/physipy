---
jupyter:
  jupytext:
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
