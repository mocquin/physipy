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

# Utils
In addition to the decorators, some utilities are available.


## numpy's arange equivalent

```python
from physipy import m
from physipy.quantity.utils import qarange
import numpy as np
```

```python
qarange(1*m, 10*m, 0.5*m)
```

## convert arrays to Quantity


Turn array of Quantity's to Quantity with array-value

```python
from physipy.quantity.utils import asqarray
```

```python
arr_of_Q = [m, 2*m, 3*m]
print(arr_of_Q)
print(asqarray(arr_of_Q))
```

Normal array will be turned to quantity

```python
dimless = asqarray(np.array([1, 2, 3]))
print(dimless)
print(type(dimless))
```

```python
dimless
```
