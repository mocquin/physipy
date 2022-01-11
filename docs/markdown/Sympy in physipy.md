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

# Dimension


Sympy is dimension is only used for printing and parsing. 


# 

```python
from physipy import m
import numpy as np
```

```python

```

```python
from physipy import m, units
mm = units["mm"]
```

```python
(mm**2).symbol
```

```python
res = np.linspace(0, 10)*m
res.symbol
```

```python

print((np.linspace(0, 10)*m).symbol)
print((np.linspace(0, 10)/m).symbol)
print((m*m).symbol)
print((m**2).symbol)
print((2*m).symbol)
print((2*m**2).symbol)
print((mm**2).symbol)
print((2*mm).symbol)

```

```python
str((2*mm).symbol)
```

```python

```
