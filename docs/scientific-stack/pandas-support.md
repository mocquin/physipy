# Pandas support

There are 2 ways `pandas` can handle `physipy`:
1. Basic support where it stores each value in a single `Quantity` : works out-of-the-box but with low performance
2. using `physipandas` : a package that takes care of the interface between `phyisipy` and `pandas`

See [physipandas](https://github.com/mocquin/physipandas) for better interface between pandas and physipy.

### Basic support out-of-the-box

Without anything else, `physipy` is kinda supported in `pandas`, but performances will be quite degraded. A 1d quantity array will be split element-wise and stored as a squence of scalar quantities, hence all operations will be done "loop"-wise, loosing the power of numpy arrays.


```python
import numpy as np
import pandas as pd

import physipy
from physipy import K, s, m, kg, units
```


```python
# create sample data as regular Quantity
arr = np.arange(10)
heights = np.random.normal(1.8, 0.1, 10)*m
heights.favunit = units["mm"]
weights = np.random.normal(74, 3, 10)*kg

print(weights)
print(heights)
```

    [68.02330891 75.29084364 78.31964794 77.95388057 75.15770254 72.83401712
     78.2871326  75.31371424 73.74678185 72.86442501] kg
    [1905.11404675 1793.50534376 1645.47961003 1752.28899466 1705.10773302
     1816.83476718 1930.41143909 1772.03471014 1764.50051484 1921.09507113] mm
    


```python
# then store Quantity in a DataFrame
df = pd.DataFrame({
    "heights":heights,
    "temp":arr*K,
    "weights":weights, 
    "arr":arr,
})
# notice the warnings below : Quantities are converted back to regular numpy arrays
# hence loosing their units.

df["heights"]
```

    C:\Users\ym\Documents\REPOS\physipy\physipy\quantity\quantity.py:753: UserWarning: The unit of the quantity is stripped for __array__
      warnings.warn(f"The unit of the quantity is stripped for {item}")
    C:\Users\ym\Documents\REPOS\physipy\physipy\quantity\quantity.py:753: UserWarning: The unit of the quantity is stripped for __array_struct__
      warnings.warn(f"The unit of the quantity is stripped for {item}")
    




    0    1.773872
    1    1.650219
    2    1.876401
    3    1.815951
    4    1.803662
    5    1.832865
    6    1.776934
    7    1.908975
    8    1.918552
    9    1.815539
    Name: heights, dtype: float64



## Full support using `physipandas`
You can also make pandas handle physipy quantities almost transparently using [`physipandas`](https://github.com/mocquin/physipandas), which is another package that extends physipy capabilities to pandas.

Previously as part of the core project, `physipandas` has been moved to its own repo in order to keep `phyisipy` as lightweight and simple as possible.
