---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Uncertainties support
As of now, basic operations are transparently handled.

Known issues : 
 - `x.nominal_value` will return the float nominal value, while you would like a quantity value (3 m, not 3). Same goes for `std_dev` and `std_score`
 - `uncertainties.umath` will fail on non-dimensionless objects, but that the case also for Quantity with physipy.math
 - there probably is a need for work on `(2*x*m+3*m).derivatives[x]` to be done
 
No array support testing has been done yet.
Also, some printing/formating testing must be done.

```python
import numpy as np
import physipy
from physipy.quantity.utils import asqarray
from physipy import m, K, s, Quantity, Dimension
import uncertainties
from uncertainties import ufloat
from uncertainties import umath
from uncertainties.umath import *  # sin(), etc.
```

```python
def info(x): print(f"{str(type(x)): <45}", " --- ", f"{str(repr(x)): <65}"+" --- "+f"{str(x): <13}")

xuv = ufloat(1.123, 0.1) 
yuv = ufloat(2.123, 0.2)
y = Quantity(ufloat(1.123, 0.1) , Dimension(None))
xuvq = xuv * s
yuvq = yuv * m
zuvq = Quantity(xuv, Dimension(None))

info(xuv)
info(y)
```

```python
info(xuv)
info(xuvq)

print("Add")
info(xuv+1)
info(xuvq+1*s)
info(xuv+xuv)
info(xuvq+xuvq)

print("Prod by int")
info(2*xuv)
info(2*xuvq)
info(xuv*2)
info(xuvq*2)

print("Product")
info(xuv*xuv)
info(xuvq*xuvq)

info(xuv * yuv)
info(xuvq * yuvq)

print("Divide by int")
info(xuv/2)
info(xuvq/2)

info(2/xuv)
info(2/xuvq)

print("Divide by other")
info(xuv/yuv)
info(xuvq/yuvq)

print("Pow by int")
info(xuv**2)
info(xuvq**2)

print("Pow by object")
info(2**xuv)
#info(2**xuvq) # TypeError: unsupported operand type(s) for ** or pow(): 'int' and 'Quantity'


print("Math functions")
info(umath.sin(xuv))
# info(umath.sin(zuvq)) # TypeError: can't convert an affine function (<class 'uncertainties.core.Variable'>) to float; use x.nominal_value
info(umath.sqrt(xuv))
# info(umath.sqrt(xuvq)) # DimensionError: Dimension error : dimension is T but should be no-dimension


print("Derivatives")
info((2*xuv+1000).derivatives[xuv])
info((2/m*xuvq+1000*s/m).derivatives[xuv]) # work to be done here


print("Attributes")
info(xuv.nominal_value)
info(xuvq.nominal_value) # needs to be wrapped with the unit
info(xuv.std_dev)
info(xuvq.std_dev) # needs to be wrapped with the unit
info(xuv.std_score(3))
info(xuvq.std_score(3)) # need to be wrapped with the unit


print("Numpy support")
# todo, not trivial as to what the expected behavior is
```

# Measurement : mix between Uncertainties and Pint
https://pint.readthedocs.io/en/stable/measurement.html?highlight=uncertainty

```python
import numpy as np
book_length = (20. * m).plus_minus(2.)
print(book_length.value)
print(2 * book_length)
```


```python

```
