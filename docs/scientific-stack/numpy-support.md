# Numpy support for arrays with dimension

A Quantity object can have any numerical-like object as its `value` attribute, including numpy's ndarray.

Physipy support numpy for many functionnalties : 
 - common creation routines
 - mathematical operations
 - numpy's functions and universal functions
 - comparison
 - indexing and fancy indexing
 - iterators


## Creation
Basic creation of dimension-full arrays : 


```python
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import m, s, Quantity, Dimension, rad
```


```python
x_samples = np.array([1, 2, 3, 4]) * m
y_samples = Quantity(np.array([1, 2, 3, 4]), Dimension("T"))
print(x_samples)
print(y_samples)
print(m*np.array([1, 2, 3, 4]) == x_samples) # multiplication is commutativ
```

    [1 2 3 4] m
    [1 2 3 4] s
    [ True  True  True  True]
    

## Operation
Basic array operation are handled the 'expected' way : note that the resulting dimension are consistent with the operation applied : 


```python
print(x_samples + 1*m)
print(x_samples * 2)
print(x_samples**2)
print(1/x_samples)
```

    [2 3 4 5] m
    [2 4 6 8] m
    [ 1  4  9 16] m**2
    [1.         0.5        0.33333333 0.25      ] 1/m
    

## Comparison

Comparison is allowed only for quantities that have the same units : 


```python
# allowed
print(x_samples > 1.5*m)

try: 
    # not allowed
    x_samples > 1.5*s
except Exception as e:
    print(e)
```

    [False  True  True  True]
    Dimension error : dimensions of operands are L and T, and are differents (length vs time).
    

## Numpy ufuncs
Most numpy ufuncs are handled the expected way, but still check for dimension correctness :


```python
q = 3*m
q_arr = np.arange(3)*m

print(np.add(q, q_arr))
print(np.multiply(q, q_arr))
print(np.sign(q_arr))
print(np.greater_equal(q_arr, 2*m))
print(np.sqrt(q_arr))
print(np.cbrt(q_arr))

print(np.cos(np.pi*rad))
print(np.tan(np.pi/4*rad))

print(np.ceil(q_arr**1.6))
print(np.negative(q_arr))
```

    [3 4 5] m
    [0 3 6] m**2
    [0 1 1]
    [False False  True]
    [0.         1.         1.41421356] m**0.5
    [0.         1.         1.25992105] m**0.333333333333333
    -1.0
    0.9999999999999999
    [0. 1. 4.] m**1.6
    [ 0 -1 -2] m
    

Trigonometric functions expect dimensionless quantities, and regular dimension correctness is expected : 


```python
try:
    np.cos(3*m)
except Exception as e:
    print(e)

try:
    np.add(3*s, q_arr)
except Exception as e:
    print(e)
```

    Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).
    Dimension error : dimensions of operands are T and L, and are differents (time vs length).
    

## Numpy's functions

Most classic numpy's functions are also handled : 


```python
print(np.linspace(3*m, 10*m, 5))
print(np.argmax(q_arr))
print(np.around(q_arr*2.3))
print(np.cross(q_arr, q_arr[::-1]))
print(np.dstack((q_arr, q_arr)))
print(np.mean(q_arr))
print(np.var(q_arr))
print(np.trapz(q_arr))
print(np.meshgrid(q_arr, q_arr))
print(np.fft.fft(q_arr))
print(np.convolve(q_arr, q_arr))
print(np.ravel(q_arr))
print(np.std(q_arr))
print(np.median(np.abs(q_arr-np.median(q_arr))))
```

    [ 3.    4.75  6.5   8.25 10.  ] m
    2 m
    [0. 2. 5.] m
    [-2  4 -2] m**2
    [[[0 0]
      [1 1]
      [2 2]]] m
    1.0 m
    0.6666666666666666 m**2
    2.0 m
    (<Quantity : [[0 1 2]
     [0 1 2]
     [0 1 2]] m>, <Quantity : [[0 0 0]
     [1 1 1]
     [2 2 2]] m>)
    [ 3. +0.j        -1.5+0.8660254j -1.5-0.8660254j] m
    [0 0 1 4 4] m**2
    [0 1 2] m
    0.816496580927726 m
    1.0 m
    

Reduce with ufuncs :


```python
import numpy as np
from physipy import m
q = np.arange(10)*m
```


```python
q = np.arange(10)*m
print(np.add.reduce(q))
print(np.multiply.reduce(q))
```

    45 m
    0 m**10
    

## Indexing

Indexing works just like with regular numpy arrays : 


```python
big_arr = np.arange(20).reshape(4,5)*s

print(big_arr)
print(big_arr[0])
print(big_arr[:, 2])
```

    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]] s
    [0 1 2 3 4] s
    [ 2  7 12 17] s
    

## Fancy indexing


```python
print(big_arr)
print(np.greater_equal(big_arr, 12*s))
print(big_arr[np.greater_equal(big_arr, 12*s)])
```

    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]] s
    [[False False False False False]
     [False False False False False]
     [False False  True  True  True]
     [ True  True  True  True  True]]
    [12 13 14 15 16 17 18 19] s
    

## Common array methods

### flat iterator


```python
print(big_arr.flat)

for q in q_arr.flat:
    print(q)
```

    <physipy.quantity.quantity.FlatQuantityIterator object at 0x0000023DCF3F98B0>
    0 m
    1 m
    2 m
    

## Known issues

### logical fucntions

The expected behavior of logical functions is not trivial : 
 - logical_and 
 - logical_or
 - logical_xor
 - logical_not
 
Hence they are not implemented.

### np.arange

The commonly used `np.arange` cannot be overriden the same way the ufuncs or classic numpy function can be. Hence, a wrapped version is provided


```python
from physipy.quantity.utils import qarange
```


```python
try:
    np.arange(10*m)
except Exception as e:
    print(e)
```

    Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).
    

    C:\Users\ym\Documents\REPOS\physipy\physipy\quantity\quantity.py:753: UserWarning: The unit of the quantity is stripped for __array_struct__
      warnings.warn(f"The unit of the quantity is stripped for {item}")
    


```python
# using range
print(np.array(range(10))*m)
# using np.arange
print(np.arange(10)*m)
# using physipy's qarange : note that the "m" quantity is inside the function call
print(qarange(10*m))
```

    [0 1 2 3 4 5 6 7 8 9] m
    [0 1 2 3 4 5 6 7 8 9] m
    [0 1 2 3 4 5 6 7 8 9] m
    

With this wrapper, you can then do the following :


```python
print(np.arange(2.5, 12)*m)
print(qarange(2.5*m, 12*m))
```

    [ 2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 10.5 11.5] m
    [ 2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 10.5 11.5] m
    

The qarange wrapper still cares about dimension correctness : 


```python
try:
    print(qarange(2*m, 10*s))
except Exception as e:
    print(e)
```

    Dimension error : dimensions of operands are L and T, and are differents (length vs time).
    


```python
np.reshape(q_arr, (1, len(q_arr)))
```




[[0. 1. 2.]]$\,m$



# List of implemented functions


```python
from physipy.quantity.quantity import HANDLED_FUNCTIONS, implemented
```


```python
set([f.__name__ for f in HANDLED_FUNCTIONS])
```




    {'amax',
     'amin',
     'append',
     'apply_along_axis',
     'argmax',
     'argmin',
     'argsort',
     'around',
     'asanyarray',
     'atleast_1d',
     'atleast_2d',
     'atleast_3d',
     'average',
     'broadcast_arrays',
     'broadcast_to',
     'clip',
     'column_stack',
     'compress',
     'concatenate',
     'convolve',
     'copy',
     'copyto',
     'corrcoef',
     'count_nonzero',
     'cov',
     'cross',
     'cumsum',
     'diagonal',
     'diff',
     'dot',
     'dstack',
     'empty_like',
     'expand_dims',
     'fft',
     'fft2',
     'fftn',
     'fftshift',
     'flip',
     'fliplr',
     'flipud',
     'full',
     'full_like',
     'gradient',
     'hfft',
     'histogram',
     'histogram2d',
     'hstack',
     'ifft',
     'ifft2',
     'ifftn',
     'ifftshift',
     'ihfft',
     'insert',
     'interp',
     'inv',
     'irfft',
     'irfft2',
     'irfftn',
     'linspace',
     'lstsq',
     'may_share_memory',
     'mean',
     'median',
     'meshgrid',
     'ndim',
     'ones_like',
     'percentile',
     'polyfit',
     'polyval',
     'prod',
     'ravel',
     'real',
     'repeat',
     'reshape',
     'rfft',
     'rfft2',
     'rfftn',
     'roll',
     'rollaxis',
     'rot90',
     'searchsorted',
     'shape',
     'sliding_window_view',
     'sort',
     'squeeze',
     'stack',
     'std',
     'sum',
     'take',
     'tile',
     'transpose',
     'trapz',
     'var',
     'vstack',
     'where',
     'zeros',
     'zeros_like'}




```python
physipy_implemented = set([f.__name__ for f in HANDLED_FUNCTIONS]).union(set(implemented))
physipy_implemented
```




    {'absolute',
     'add',
     'amax',
     'amin',
     'append',
     'apply_along_axis',
     'arccos',
     'arccosh',
     'arcsin',
     'arcsinh',
     'arctan',
     'arctan2',
     'arctanh',
     'argmax',
     'argmin',
     'argsort',
     'around',
     'asanyarray',
     'atleast_1d',
     'atleast_2d',
     'atleast_3d',
     'average',
     'broadcast_arrays',
     'broadcast_to',
     'cbrt',
     'ceil',
     'clip',
     'column_stack',
     'compress',
     'concatenate',
     'conj',
     'conjugate',
     'convolve',
     'copy',
     'copysign',
     'copyto',
     'corrcoef',
     'cos',
     'cosh',
     'count_nonzero',
     'cov',
     'cross',
     'cumsum',
     'deg2rad',
     'diagonal',
     'diff',
     'divide',
     'dot',
     'dstack',
     'empty_like',
     'equal',
     'exp',
     'exp2',
     'expand_dims',
     'expm1',
     'fabs',
     'fft',
     'fft2',
     'fftn',
     'fftshift',
     'flip',
     'fliplr',
     'flipud',
     'floor',
     'floor_divide',
     'fmax',
     'fmin',
     'fmod',
     'full',
     'full_like',
     'gradient',
     'greater',
     'greater_equal',
     'hfft',
     'histogram',
     'histogram2d',
     'hstack',
     'hypot',
     'ifft',
     'ifft2',
     'ifftn',
     'ifftshift',
     'ihfft',
     'insert',
     'interp',
     'inv',
     'irfft',
     'irfft2',
     'irfftn',
     'isfinite',
     'isinf',
     'isnan',
     'less',
     'less_equal',
     'linspace',
     'log',
     'log10',
     'log1p',
     'log2',
     'logaddexp',
     'logaddexp2',
     'lstsq',
     'matmul',
     'maximum',
     'may_share_memory',
     'mean',
     'median',
     'meshgrid',
     'minimum',
     'mod',
     'modf',
     'multiply',
     'ndim',
     'negative',
     'nextafter',
     'not_equal',
     'ones_like',
     'percentile',
     'polyfit',
     'polyval',
     'power',
     'prod',
     'rad2deg',
     'ravel',
     'real',
     'reciprocal',
     'remainder',
     'repeat',
     'reshape',
     'rfft',
     'rfft2',
     'rfftn',
     'rint',
     'roll',
     'rollaxis',
     'rot90',
     'searchsorted',
     'shape',
     'sign',
     'sin',
     'sinh',
     'sliding_window_view',
     'sort',
     'sqrt',
     'square',
     'squeeze',
     'stack',
     'std',
     'subtract',
     'sum',
     'take',
     'tan',
     'tanh',
     'tile',
     'transpose',
     'trapz',
     'true_divide',
     'trunc',
     'var',
     'vstack',
     'where',
     'zeros',
     'zeros_like'}



# List of not implemented functions

From https://github.com/hgrecco/pint/commit/2da1be75878e6da53f658b79ed057cc0b34b8c05


```python
import numpy as np

numpy_functions = set(attr for attr in dir(np) if hasattr(getattr(np, attr), '_implementation'))

print(sorted(numpy_functions - physipy_implemented))
```

    ['alen', 'all', 'allclose', 'alltrue', 'angle', 'any', 'apply_over_axes', 'argpartition', 'argwhere', 'array2string', 'array_equal', 'array_equiv', 'array_repr', 'array_split', 'array_str', 'asfarray', 'asscalar', 'bincount', 'block', 'busday_count', 'busday_offset', 'can_cast', 'choose', 'common_type', 'correlate', 'cumprod', 'cumproduct', 'datetime_as_string', 'delete', 'diag', 'diag_indices_from', 'diagflat', 'digitize', 'dsplit', 'ediff1d', 'einsum', 'einsum_path', 'extract', 'fill_diagonal', 'fix', 'flatnonzero', 'geomspace', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'i0', 'imag', 'in1d', 'inner', 'intersect1d', 'is_busday', 'isclose', 'iscomplex', 'iscomplexobj', 'isin', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'ix_', 'kron', 'lexsort', 'logspace', 'max', 'min', 'min_scalar_type', 'moveaxis', 'msort', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nonzero', 'outer', 'packbits', 'pad', 'partition', 'piecewise', 'place', 'poly', 'polyadd', 'polyder', 'polydiv', 'polyint', 'polymul', 'polysub', 'product', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'ravel_multi_index', 'real_if_close', 'resize', 'result_type', 'roots', 'round', 'round_', 'row_stack', 'save', 'savetxt', 'savez', 'savez_compressed', 'select', 'setdiff1d', 'setxor1d', 'shares_memory', 'sinc', 'size', 'sometrue', 'sort_complex', 'split', 'swapaxes', 'take_along_axis', 'tensordot', 'trace', 'tril', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices_from', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unwrap', 'vander', 'vdot', 'vsplit']
    

# Proxy support for numpy.random functions


```python
from physipy import calculus, s
import numpy as np
```

For now you have to manually create random vectors since numpy's random functions do not support interface :


```python
np.random.normal(1, 2, 10000)*s
```




[ 4.08696168  0.92381521  4.5491531  ...  3.45169288 -0.19269588
  2.48078724]$\,s$


