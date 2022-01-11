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

# Quantity


## Construction from class


Several ways are available constructing a Quantity object

```python
from physipy import Quantity, m, sr, Dimension
```

Scalar Quantity

```python
# from int
print(Quantity(1, Dimension("L")))

# from float
print(Quantity(1.123, Dimension("L")))

# from fraction.Fraction
from fractions import Fraction
print(Quantity(Fraction(1, 2), Dimension("L")))
```


Array-like Quantity

```python
# from list
print(Quantity([1, 2, 3, 4], Dimension("L")))

# from tuple
print(Quantity((1, 2, 3, 4), Dimension("L")))

# from np.ndarray
import numpy as np
print(Quantity(np.array([1, 2, 3, 4]), Dimension("L")))
```

## Construction by multiplicating value with unit/quantity


Because the multiplication of quantity first tries to "quantify" the other operand, several creation routines by multicpliation are available

```python
# multiplicating int
print(1 * m)

# multiplicating float
print(1.123 * m)

# multiplicating Fraction
print(Fraction(1, 2) * m)

# multiplicating list
print([1, 2, 3, 4] * m)

# multiplicating list
print((1, 2, 3, 4) * m)

# multiplicating array
print(np.array([1, 2, 3, 4]) * m)
```

# Known issues


## Quantity defition with minus sign

```python
from physipy import Quantity, Dimension

print(type(-Quantity(5, Dimension(None)))) # returns int
print(type(Quantity(-5, Dimension(None)))) # returns Quantity
print(type(Quantity(5, Dimension(None)))) # returns Quantity
```

```python

```
