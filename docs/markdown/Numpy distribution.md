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
from physipy import random as phyrandom, s
import numpy as np
```

```python
np.random.normal(1, 2, 10000)*s
```

```python
print(phyrandom.normal(1, 2, 10000))
print(phyrandom.normal(1, 2, 10000)*s)
print(phyrandom.normal(1*s, 2*s, 10000))
```
