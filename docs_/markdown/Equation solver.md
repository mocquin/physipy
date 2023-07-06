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

# Using scipy wrappers


Scipy offers various solver algorithms in `scipy.optimize`. Some of the solvers are wrapped and presented below.


## Root solver


A wrapper of `scipy.optimize.root`:

```python
from physipy import s
from physipy.optimize import root

def toto(t):
    return -10*s + t
```

```python
print(root(toto, 0*s))
```

```python
def tata(t, p):
    return -10*s*p + t

print(root(tata, 0*s, args=(0.5,)))
```

### Quadratic Brent method


A wrapper of `scipy.optimize.brentq`:

```python
from physipy.optimize import brentq
```


```python
print(brentq(toto, -10*s, 10*s))
print(brentq(tata, -10*s, 10*s, args=(0.5,)))
```


```python

```
