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

<!-- #region slideshow={"slide_type": "slide"} tags=[] -->
# Pandas support
<!-- #endregion -->

Without anything else, physipy is kinda supported in pandas, but performances will be quite degraded. It seems a 1d quantity array will be split element-wise and stored, hence all operations will be done "loop"-wise, loosing the power of numpy arrays.

See [physipandas](https://github.com/mocquin/physipandas) for better interface between pandas and physipy.

```python
import numpy as np
import pandas as pd

import physipy
from physipy import K, s, m, kg, units
```

```python
arr = np.arange(10)
heights = np.random.normal(1.8, 0.1, 10)*m
heights.favunit = units["mm"]
weights = np.random.normal(74, 3, 10)*kg
heights
```

<!-- #region slideshow={"slide_type": "slide"} tags=[] -->
## Dataframe
<!-- #endregion -->

```python
df = pd.DataFrame({
    "heights":heights,
    "temp":arr*K,
    "weights":weights, 
    "arr":arr,
})
```

```python
df["heights"]
```

```python
df
```

```python
df.info()
```

```python
df.describe()
```

```python
%timeit df["heights"].min()
```

```python
%timeit df["arr"].min()
```
