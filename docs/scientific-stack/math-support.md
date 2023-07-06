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

# Python's math module

```python
from physipy import m, rad, Quantity, Dimension
import physipy
import physipy.math as phymath
```

```python
import math
from math import acos, acosh, asin, asinh, atan, atan2, atanh
from math import ceil, copysign, comb, cos, cosh
from math import degrees, dist
from math import erf, erfc, exp, expm1
from math import fabs, factorial, floor, fmod, frexp, fsum
from math import gamma, gcd
from math import hypot
from math import isclose, isfinite, isinf, isnan, isqrt
from math import ldexp, lgamma, log, log10, log1p, log2
from math import modf
from math import perm, pow as math_pow, prod
from math import radians, remainder
from math import sin, sinh, sqrt
from math import tan, tanh, trunc

# from math import nextafter, ulp, lcm

a = 5.123 * m
b = -2*m
```

```python
math_params = {
    "acos"     :(acos     , a       ),
    "acosh"    :(acosh    , a       ),
    "asin"     :(asin     , a       ),
    "asinh"    :(asinh    , a       ),
    "atan"     :(atan     , a       ),
    "atan2"    :(atan2    , (a,b)   ),
    "atanh"    :(atanh    , a       ),
    "ceil"     :(ceil     , a       ),
    "coysign"  :(copysign , (a, b)  ),
    #"comb"     :(comb     , 10, 3),  
    "cos"      :(cos      , a       ),
    "cosh"     :(cosh     , a       ),
    "degrees"  :(degrees  , a       ),
    "dist"     :(dist     , ([1*m, 3*m], [2*m, 5*m])),
    "erf"      :(erf      , a       ),
    "erfc"     :(erfc     , a       ),
    "exp"      :(exp      , a       ),
    "expm1"    :(expm1    , a       ),
    "fabs"     :(fabs     , a       ),
    "floor"    :(floor    , a       ),
    "fmod"     :(fmod     , (a, b)  ),
    "fsum"     :(fsum     , [a, a, a]),
    "gamma"    :(gamma    , a       ),
    "gcd"      :(gcd      , (a, a)  ),
    "hypot"    :(hypot    , a       ),
    "isclose"  :(isclose  , (a, b)  ),
    "isfinite" :(isfinite , a       ),
    "isinf"    :(isinf    , a       ), 
    "isnan"    :(isnan    , a       ), 
    "isqrt"    :(isqrt    , a       ),
    "ldexp"    :(ldexp    , (a, a)  ), 
    "lgamma"   :(lgamma   , a       ),
    "log"      :(log      , a       ),
    "log10"    :(log10    , a       ),
    "log1p"    :(log1p    , a       ),
    "log2"     :(log2     , a       ),
    "modf"     :(modf     , a       ),
    "perm"     :(perm     , a       ),
    "pow"      :(math_pow , (a, 2)  ),
    "prod"     :(prod     , [a, a]  ),
    "radians"  :(radians  , a       ),
    "remainder":(remainder, (a, b)  ),
    "sin"      :(sin      , a       ), 
    "sinh"     :(sinh     , a       ), 
    "sqrt"     :(sqrt     , a       ),
    "tan"      :(tan      , a       ), 
    "tanh"     :(tanh     , a       ),
    "trunc"    :(trunc    , a       ),
}


q_rad = 0.4*rad
q = 2.123*m
q_dimless = Quantity(1, Dimension(None))

physipy_math_params = {
    "acos"     :(physipy.math.acos     , q_dimless       ),
    "acosh"    :(physipy.math.acosh    , q_dimless       ),
    "asin"     :(physipy.math.asin     , q_dimless       ),
    "asinh"    :(physipy.math.asinh    , q_dimless       ),
    "atan"     :(physipy.math.atan     , q_dimless       ),
    "atan2"    :(physipy.math.atan2    , (q,q)   ),
    "atanh"    :(physipy.math.atanh    , q_dimless       ),
    "ceil"     :(physipy.math.ceil     , q       ),
    "coysign"  :(physipy.math.copysign , (q, q)  ),
    #"comb"     :(comb     , 10, 3),  
    "cos"      :(physipy.math.cos      , q       ),
    "cosh"     :(physipy.math.cosh     , q       ),
    "degrees"  :(physipy.math.degrees  , q       ),
    "dist"     :(physipy.math.dist     , ([q, q], [q, q])),
    "erf"      :(physipy.math.erf      , q       ),
    "erfc"     :(physipy.math.erfc     , q       ),
    "exp"      :(physipy.math.exp      , q       ),
    "expm1"    :(physipy.math.expm1    , q       ),
    "fabs"     :(physipy.math.fabs     , q       ),
    "floor"    :(physipy.math.floor    , q       ),
    "fmod"     :(physipy.math.fmod     , (q, q)  ),
    "fsum"     :(physipy.math.fsum     , [q, q, q]),
    "gamma"    :(physipy.math.gamma    , q       ),
    "gcd"      :(physipy.math.gcd      , (q, q)  ),
    "hypot"    :(physipy.math.hypot    ,q       ),
    "isclose"  :(physipy.math.isclose  , (a, b)  ),
    "isfinite" :(physipy.math.isfinite ,q       ),
    "isinf"    :(physipy.math.isinf    ,q       ), 
    "isnan"    :(physipy.math.isnan    ,q       ), 
    "isqrt"    :(physipy.math.isqrt    ,q       ),
    "ldexp"    :(physipy.math.ldexp    , (q,q)  ), 
    "lgamma"   :(physipy.math.lgamma   ,q_dimless       ),
    "log"      :(physipy.math.log      ,q_dimless       ),
    "log10"    :(physipy.math.log10    ,q_dimless       ),
    "log1p"    :(physipy.math.log1p    ,q_dimless       ),
    "log2"     :(physipy.math.log2     ,q_dimless       ),
    "modf"     :(physipy.math.modf     ,q       ),
    "perm"     :(physipy.math.perm     ,q       ),
    "pow"      :(physipy.math.pow , (q, 2)  ),
    "prod"     :(physipy.math.prod     , [q,q]  ),
    "radians"  :(physipy.math.radians  ,q       ),
    "remainder":(physipy.math.remainder, (q, b)  ),
    "sin"      :(physipy.math.sin      ,q       ), 
    "sinh"     :(physipy.math.sinh     ,q       ), 
    "sqrt"     :(physipy.math.sqrt     ,q       ),
    "tan"      :(physipy.math.tan      ,q       ), 
    "tanh"     :(physipy.math.tanh     ,q       ),
    "trunc"    :(physipy.math.trunc    ,q       ),
}
```

```python
import time
class Timer():
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
def color_green_true_red_false(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'green' if val else 'red'
    return 'color: %s' % color
```

```python
res = {}

for name, func_and_args in math_params.items():
    func, args = func_and_args
    results = {}
    try:
        results["Input"] = ", ".join([str(i) for i in args])
    except:
        results["Input"] = str(args)
    if type(args) is tuple:
        # For math
        try:
            with Timer() as timer:
                v = func(*args)
            results["Passed"] = True
            results["Returned"] = str(v)    
            results["Time"] = timer.msecs
        except Exception as e:
            results["Passed"] = False
            results["Returned"] = str(e)   
            results["Time"] = None
    else:
        try:
            with Timer() as timer:
                v = func(args)
            results["Passed"] = True
            results["Returned"] = str(v)  
            results["Time"] = timer.msecs
        except Exception as e:
            results["Passed"] = False
            results["Returned"] = str(e)  
            results["Time"] = None
    res[name] = results
```

```python
res_phymath = {}

for name, func_and_args in physipy_math_params.items():
    func, args = func_and_args
    results = {}
    try:
        results["Input"] = ", ".join([str(i) for i in args])
    except:
        results["Input"] = str(args)
    if type(args) is tuple:
        # For math
        try:
            with Timer() as timer:
                v = getattr(phymath, name)(*args)
            results["Passed"] = True
            results["Returned"] = str(v)    
            results["Time"] = timer.msecs
        except Exception as e:
            results["Passed"] = False
            results["Returned"] = str(e)   
            results["Time"] = None
    else:
        try:
            with Timer() as timer:
                v = getattr(phymath, name)(args)
            results["Passed"] = True
            results["Returned"] = str(v)  
            results["Time"] = timer.msecs
        except Exception as e:
            results["Passed"] = False
            results["Returned"] = str(e)  
            results["Time"] = None
    res_phymath[name] = results
```

```python
import pandas as pd
df = pd.DataFrame.from_dict(res, orient="index")
df = df.style.applymap(color_green_true_red_false, subset=pd.IndexSlice[:, ['Passed']])
df_phymath = pd.DataFrame.from_dict(res_phymath, orient="index")
df_phymath = df_phymath.style.applymap(color_green_true_red_false, subset=pd.IndexSlice[:, ['Passed']])
```

```python
df
```

```python
df_phymath
```

```python

```
