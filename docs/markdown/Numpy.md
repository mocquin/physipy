---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Numpy support for arrays with dimension


A Quantity object can have any numerical-like object as its `value` attribute, including numpy's ndarray.


Physipy support numpy for many functionnalties : 
 - common creation routines
 - mathematical operations
 - numpy's functions and universal functions
 - comparison
 - indexing and fancy indexing
 - iterators



## Creation
Basic creation of dimension-full arrays : 

```python
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import physipy
from physipy import m, s, Quantity, Dimension, rad
```

```python
x_samples = np.array([1, 2, 3, 4]) * m
y_samples = Quantity(np.array([1, 2, 3, 4]), Dimension("T"))
print(x_samples)
print(y_samples)
print(m*np.array([1, 2, 3, 4]) == x_samples) # multiplication is commutativ
```

## Operation
Basic array operation are handled the 'expected' way : note that the resulting dimension are consistent with the operation applied : 

```python
print(x_samples + 1*m)
print(x_samples * 2)
print(x_samples**2)
print(1/x_samples)

```

## Comparison

```python
print(x_samples > 1.5*m)

try: 
    x_samples > 1.5*s
except Exception as e:
    print(e)

```

## Numpy ufuncs
Most numpy ufuncs are handled the expected way, but still check for dimension correctness :

```python
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
```

Trigonometric functions expect dimensionless quantities, and regular dimension correctness is expected : 

```python
try:
    np.cos(3*m)
except Exception as e:
    print(e)

try:
    np.add(3*s, q_arr)
except Exception as e:
    print(e)
```

## Numpy's functions


Most classic numpy's functions are also handled : 

```python
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
```

Reduce with ufuncs :

```python
import numpy as np
from physipy import m
q = np.arange(10)*m
```

```python
q = np.arange(10)*m
print(np.add.reduce(q))
print(np.multiply.reduce(q))
```

## Indexing

```python
big_arr = np.arange(20).reshape(4,5)*s

print(big_arr)
print(big_arr[0])
print(big_arr[:, 2])
```

## Fancy indexing

```python
print(big_arr)
print(np.greater_equal(big_arr, 12*s))
print(big_arr[np.greater_equal(big_arr, 12*s)])
```

## Common array methods


### flat iterator

```python
print(big_arr.flat)

for q in q_arr.flat:
    print(q)

```

## Known issues


### logical fucntions


The expected behavior of logical functions is not trivial : 
 - logical_and 
 - logical_or
 - logical_xor
 - logical_not
 
Hence they are not implemented.


### np.arange


The commonly used `np.arange` cannot be overriden the same way the ufuncs or classic numpy function can be. Hence, a wrapped version is provided

```python
from physipy.quantity.utils import qarange
```

```python
# using range
print(np.array(range(10))*m)
# using np.arange
print(np.arange(10)*m)
# using physipy's qarange : note that the "m" quantity is inside the function call
print(qarange(10*m))
```

With this wrapper, you can then do the following :

```python
print(np.arange(2.5, 12)*m)
print(qarange(2.5*m, 12*m))
```

The qarange wrapper still cares about dimension correctness : 

```python
try:
    print(qarange(2*m, 10*s))
except Exception as e:
    print(e)
```

```python
np.reshape(q_arr, (1, len(q_arr)))
```

# List of implemented functions

```python
from physipy.quantity.quantity import HANDLED_FUNCTIONS, implemented

physipy_implemented = set([f.__name__ for f in HANDLED_FUNCTIONS]).union(set(implemented))
physipy_implemented
```

<!-- #region tags=[] -->
# List of not implemented functions
<!-- #endregion -->

From https://github.com/hgrecco/pint/commit/2da1be75878e6da53f658b79ed057cc0b34b8c05

```python
import numpy as np

numpy_functions = set(attr for attr in dir(np) if hasattr(getattr(np, attr), '_implementation'))

print(sorted(numpy_functions - physipy_implemented))
```
