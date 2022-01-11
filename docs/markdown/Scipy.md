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
