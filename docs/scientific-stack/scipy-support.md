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

# Scipy.spatial.distance


Scipy first casts arrays into numpy array so no dimension checking is done.

```python
import scipy.spatial.distance
from  scipy.spatial.distance import squareform
import numpy as np
from physipy import K, m, s, units

```

```python
nd_space = 20
m_obs = 4
arr = np.random.randn(m_obs, nd_space) * m

# returned matrix will be of shape m_obs x m_obs
print(squareform(scipy.spatial.distance.pdist(arr.value, "euclidean")))         # same dim
print(squareform(scipy.spatial.distance.pdist(arr.value, "minkowski", p=2)))    # same dim, # same as euclidean for p=2
print(squareform(scipy.spatial.distance.pdist(arr.value, 'cityblock')))         # same dim
print(squareform(scipy.spatial.distance.pdist(arr.value, 'seuclidean')))        # same dim
print(squareform(scipy.spatial.distance.pdist(arr.value, 'sqeuclidean')))       # same dim
print(squareform(scipy.spatial.distance.pdist(arr.value, 'cosine')))            # any
print(squareform(scipy.spatial.distance.pdist(arr.value, 'correlation')))       # any
print(squareform(scipy.spatial.distance.pdist(arr.value, 'hamming')))           # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'chebyshev')))         # same dim 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'canberra')))          # same dim 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'braycurtis')))        # same dim 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'yule')))              # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'dice')))              # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'kulsinski')))         # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'rogerstanimoto')))    # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'russellrao')))        # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'sokalmichener')))     # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'sokalsneath')))       # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'kulczynski1')))       # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, 'sokalmichener')))     # bool 
print(squareform(scipy.spatial.distance.pdist(arr.value, lambda u, v: np.sqrt(((u-v)**2).sum()))))
```

```python

```

```python
import scipy
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

from physipy import K, m, s, units
```

```python
# some model to fit
def model(time, gain):
    return 2*time*gain
```

```python
x_time = np.array([1, 2, 3, 4, 5])*s
true_gain = 1.2*m/s

noise = np.random.randn(x_time.shape[0])
true_y = 2*x_time*true_gain
y_height = true_y + noise*m
```


```python
fig, ax = plt.subplots()
ax.plot(x_time, true_y, label="True y")
ax.plot(x_time, noise, label="Noise")
ax.plot(x_time, y_height, label="Data y")
plt.legend()
```


```python
optimize.curve_fit(model, x_time.value, y_height.value, p0=[(1*m/s).value])
```

Must use an initial guess, otherwise it tries a float and fail the dimension comparisons.
With an initial guess, "ValueError: setting an array element with a sequence"

```python
print(np.atleast_1d(x_time))
print(np.atleast_1d(y_height))
print(np.atleast_1d([1*m/s]))
print(np.asanyarray([1*m/s]))
```

```python

```


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
