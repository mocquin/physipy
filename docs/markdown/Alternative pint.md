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

```python
import pint
import numpy as np
```

```python
ureg = pint.UnitRegistry()
```

```python
iter(2*ureg.m)
```

```python
np.full((3, 3), 2*ureg.m)
```

```python
a = np.array([1, 2, 3]) * ureg.m
b = np.array([1, 1, 1]) * ureg.m
np.copyto(a, b)
print(a)
print(b)
```

```python
a = np.array([1, 2, 3]) * ureg.m
b = np.array([1, 1, 1]) 
np.copyto(a, b)
print(a)
print(b)
```

```python
a = np.array([1, 2, 3])
b = np.array([1, 1, 1]) * ureg.m
np.copyto(a, b)
print(a)
print(b)
```

# from array(2m)

```python
np.array(2*ureg.m)
```

```python

```
