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

    C:\Users\ym\Documents\REPOS\physipy\physipy\quantity\quantity.py:753: UserWarning: The unit of the quantity is stripped for __array_struct__
      warnings.warn(f"The unit of the quantity is stripped for {item}")
    


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




<style type="text/css">
#T_3d8fc_row0_col1, #T_3d8fc_row1_col1, #T_3d8fc_row2_col1, #T_3d8fc_row3_col1, #T_3d8fc_row4_col1, #T_3d8fc_row5_col1, #T_3d8fc_row6_col1, #T_3d8fc_row8_col1, #T_3d8fc_row9_col1, #T_3d8fc_row10_col1, #T_3d8fc_row11_col1, #T_3d8fc_row12_col1, #T_3d8fc_row13_col1, #T_3d8fc_row14_col1, #T_3d8fc_row15_col1, #T_3d8fc_row16_col1, #T_3d8fc_row17_col1, #T_3d8fc_row19_col1, #T_3d8fc_row20_col1, #T_3d8fc_row21_col1, #T_3d8fc_row22_col1, #T_3d8fc_row23_col1, #T_3d8fc_row24_col1, #T_3d8fc_row25_col1, #T_3d8fc_row26_col1, #T_3d8fc_row27_col1, #T_3d8fc_row28_col1, #T_3d8fc_row29_col1, #T_3d8fc_row30_col1, #T_3d8fc_row31_col1, #T_3d8fc_row32_col1, #T_3d8fc_row33_col1, #T_3d8fc_row34_col1, #T_3d8fc_row35_col1, #T_3d8fc_row36_col1, #T_3d8fc_row37_col1, #T_3d8fc_row39_col1, #T_3d8fc_row40_col1, #T_3d8fc_row41_col1, #T_3d8fc_row42_col1, #T_3d8fc_row43_col1, #T_3d8fc_row44_col1, #T_3d8fc_row45_col1 {
  color: red;
}
#T_3d8fc_row7_col1, #T_3d8fc_row18_col1, #T_3d8fc_row38_col1, #T_3d8fc_row46_col1 {
  color: green;
}
</style>
<table id="T_3d8fc">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3d8fc_level0_col0" class="col_heading level0 col0" >Input</th>
      <th id="T_3d8fc_level0_col1" class="col_heading level0 col1" >Passed</th>
      <th id="T_3d8fc_level0_col2" class="col_heading level0 col2" >Returned</th>
      <th id="T_3d8fc_level0_col3" class="col_heading level0 col3" >Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3d8fc_level0_row0" class="row_heading level0 row0" >acos</th>
      <td id="T_3d8fc_row0_col0" class="data row0 col0" >5.123 m</td>
      <td id="T_3d8fc_row0_col1" class="data row0 col1" >False</td>
      <td id="T_3d8fc_row0_col2" class="data row0 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row0_col3" class="data row0 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row1" class="row_heading level0 row1" >acosh</th>
      <td id="T_3d8fc_row1_col0" class="data row1 col0" >5.123 m</td>
      <td id="T_3d8fc_row1_col1" class="data row1 col1" >False</td>
      <td id="T_3d8fc_row1_col2" class="data row1 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row1_col3" class="data row1 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row2" class="row_heading level0 row2" >asin</th>
      <td id="T_3d8fc_row2_col0" class="data row2 col0" >5.123 m</td>
      <td id="T_3d8fc_row2_col1" class="data row2 col1" >False</td>
      <td id="T_3d8fc_row2_col2" class="data row2 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row2_col3" class="data row2 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row3" class="row_heading level0 row3" >asinh</th>
      <td id="T_3d8fc_row3_col0" class="data row3 col0" >5.123 m</td>
      <td id="T_3d8fc_row3_col1" class="data row3 col1" >False</td>
      <td id="T_3d8fc_row3_col2" class="data row3 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row3_col3" class="data row3 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row4" class="row_heading level0 row4" >atan</th>
      <td id="T_3d8fc_row4_col0" class="data row4 col0" >5.123 m</td>
      <td id="T_3d8fc_row4_col1" class="data row4 col1" >False</td>
      <td id="T_3d8fc_row4_col2" class="data row4 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row4_col3" class="data row4 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row5" class="row_heading level0 row5" >atan2</th>
      <td id="T_3d8fc_row5_col0" class="data row5 col0" >5.123 m, -2 m</td>
      <td id="T_3d8fc_row5_col1" class="data row5 col1" >False</td>
      <td id="T_3d8fc_row5_col2" class="data row5 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row5_col3" class="data row5 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row6" class="row_heading level0 row6" >atanh</th>
      <td id="T_3d8fc_row6_col0" class="data row6 col0" >5.123 m</td>
      <td id="T_3d8fc_row6_col1" class="data row6 col1" >False</td>
      <td id="T_3d8fc_row6_col2" class="data row6 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row6_col3" class="data row6 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row7" class="row_heading level0 row7" >ceil</th>
      <td id="T_3d8fc_row7_col0" class="data row7 col0" >5.123 m</td>
      <td id="T_3d8fc_row7_col1" class="data row7 col1" >True</td>
      <td id="T_3d8fc_row7_col2" class="data row7 col2" >6 m</td>
      <td id="T_3d8fc_row7_col3" class="data row7 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row8" class="row_heading level0 row8" >coysign</th>
      <td id="T_3d8fc_row8_col0" class="data row8 col0" >5.123 m, -2 m</td>
      <td id="T_3d8fc_row8_col1" class="data row8 col1" >False</td>
      <td id="T_3d8fc_row8_col2" class="data row8 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row8_col3" class="data row8 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row9" class="row_heading level0 row9" >cos</th>
      <td id="T_3d8fc_row9_col0" class="data row9 col0" >5.123 m</td>
      <td id="T_3d8fc_row9_col1" class="data row9 col1" >False</td>
      <td id="T_3d8fc_row9_col2" class="data row9 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row9_col3" class="data row9 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row10" class="row_heading level0 row10" >cosh</th>
      <td id="T_3d8fc_row10_col0" class="data row10 col0" >5.123 m</td>
      <td id="T_3d8fc_row10_col1" class="data row10 col1" >False</td>
      <td id="T_3d8fc_row10_col2" class="data row10 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row10_col3" class="data row10 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row11" class="row_heading level0 row11" >degrees</th>
      <td id="T_3d8fc_row11_col0" class="data row11 col0" >5.123 m</td>
      <td id="T_3d8fc_row11_col1" class="data row11 col1" >False</td>
      <td id="T_3d8fc_row11_col2" class="data row11 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row11_col3" class="data row11 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row12" class="row_heading level0 row12" >dist</th>
      <td id="T_3d8fc_row12_col0" class="data row12 col0" >[<Quantity : 1 m, symbol=m*UndefinedSymbol>, <Quantity : 3 m, symbol=m*UndefinedSymbol>], [<Quantity : 2 m, symbol=m*UndefinedSymbol>, <Quantity : 5 m, symbol=m*UndefinedSymbol>]</td>
      <td id="T_3d8fc_row12_col1" class="data row12 col1" >False</td>
      <td id="T_3d8fc_row12_col2" class="data row12 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row12_col3" class="data row12 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row13" class="row_heading level0 row13" >erf</th>
      <td id="T_3d8fc_row13_col0" class="data row13 col0" >5.123 m</td>
      <td id="T_3d8fc_row13_col1" class="data row13 col1" >False</td>
      <td id="T_3d8fc_row13_col2" class="data row13 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row13_col3" class="data row13 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row14" class="row_heading level0 row14" >erfc</th>
      <td id="T_3d8fc_row14_col0" class="data row14 col0" >5.123 m</td>
      <td id="T_3d8fc_row14_col1" class="data row14 col1" >False</td>
      <td id="T_3d8fc_row14_col2" class="data row14 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row14_col3" class="data row14 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row15" class="row_heading level0 row15" >exp</th>
      <td id="T_3d8fc_row15_col0" class="data row15 col0" >5.123 m</td>
      <td id="T_3d8fc_row15_col1" class="data row15 col1" >False</td>
      <td id="T_3d8fc_row15_col2" class="data row15 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row15_col3" class="data row15 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row16" class="row_heading level0 row16" >expm1</th>
      <td id="T_3d8fc_row16_col0" class="data row16 col0" >5.123 m</td>
      <td id="T_3d8fc_row16_col1" class="data row16 col1" >False</td>
      <td id="T_3d8fc_row16_col2" class="data row16 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row16_col3" class="data row16 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row17" class="row_heading level0 row17" >fabs</th>
      <td id="T_3d8fc_row17_col0" class="data row17 col0" >5.123 m</td>
      <td id="T_3d8fc_row17_col1" class="data row17 col1" >False</td>
      <td id="T_3d8fc_row17_col2" class="data row17 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row17_col3" class="data row17 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row18" class="row_heading level0 row18" >floor</th>
      <td id="T_3d8fc_row18_col0" class="data row18 col0" >5.123 m</td>
      <td id="T_3d8fc_row18_col1" class="data row18 col1" >True</td>
      <td id="T_3d8fc_row18_col2" class="data row18 col2" >5 m</td>
      <td id="T_3d8fc_row18_col3" class="data row18 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row19" class="row_heading level0 row19" >fmod</th>
      <td id="T_3d8fc_row19_col0" class="data row19 col0" >5.123 m, -2 m</td>
      <td id="T_3d8fc_row19_col1" class="data row19 col1" >False</td>
      <td id="T_3d8fc_row19_col2" class="data row19 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row19_col3" class="data row19 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row20" class="row_heading level0 row20" >fsum</th>
      <td id="T_3d8fc_row20_col0" class="data row20 col0" >5.123 m, 5.123 m, 5.123 m</td>
      <td id="T_3d8fc_row20_col1" class="data row20 col1" >False</td>
      <td id="T_3d8fc_row20_col2" class="data row20 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row20_col3" class="data row20 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row21" class="row_heading level0 row21" >gamma</th>
      <td id="T_3d8fc_row21_col0" class="data row21 col0" >5.123 m</td>
      <td id="T_3d8fc_row21_col1" class="data row21 col1" >False</td>
      <td id="T_3d8fc_row21_col2" class="data row21 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row21_col3" class="data row21 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row22" class="row_heading level0 row22" >gcd</th>
      <td id="T_3d8fc_row22_col0" class="data row22 col0" >5.123 m, 5.123 m</td>
      <td id="T_3d8fc_row22_col1" class="data row22 col1" >False</td>
      <td id="T_3d8fc_row22_col2" class="data row22 col2" >'Quantity' object cannot be interpreted as an integer</td>
      <td id="T_3d8fc_row22_col3" class="data row22 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row23" class="row_heading level0 row23" >hypot</th>
      <td id="T_3d8fc_row23_col0" class="data row23 col0" >5.123 m</td>
      <td id="T_3d8fc_row23_col1" class="data row23 col1" >False</td>
      <td id="T_3d8fc_row23_col2" class="data row23 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row23_col3" class="data row23 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row24" class="row_heading level0 row24" >isclose</th>
      <td id="T_3d8fc_row24_col0" class="data row24 col0" >5.123 m, -2 m</td>
      <td id="T_3d8fc_row24_col1" class="data row24 col1" >False</td>
      <td id="T_3d8fc_row24_col2" class="data row24 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row24_col3" class="data row24 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row25" class="row_heading level0 row25" >isfinite</th>
      <td id="T_3d8fc_row25_col0" class="data row25 col0" >5.123 m</td>
      <td id="T_3d8fc_row25_col1" class="data row25 col1" >False</td>
      <td id="T_3d8fc_row25_col2" class="data row25 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row25_col3" class="data row25 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row26" class="row_heading level0 row26" >isinf</th>
      <td id="T_3d8fc_row26_col0" class="data row26 col0" >5.123 m</td>
      <td id="T_3d8fc_row26_col1" class="data row26 col1" >False</td>
      <td id="T_3d8fc_row26_col2" class="data row26 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row26_col3" class="data row26 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row27" class="row_heading level0 row27" >isnan</th>
      <td id="T_3d8fc_row27_col0" class="data row27 col0" >5.123 m</td>
      <td id="T_3d8fc_row27_col1" class="data row27 col1" >False</td>
      <td id="T_3d8fc_row27_col2" class="data row27 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row27_col3" class="data row27 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row28" class="row_heading level0 row28" >isqrt</th>
      <td id="T_3d8fc_row28_col0" class="data row28 col0" >5.123 m</td>
      <td id="T_3d8fc_row28_col1" class="data row28 col1" >False</td>
      <td id="T_3d8fc_row28_col2" class="data row28 col2" >'Quantity' object cannot be interpreted as an integer</td>
      <td id="T_3d8fc_row28_col3" class="data row28 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row29" class="row_heading level0 row29" >ldexp</th>
      <td id="T_3d8fc_row29_col0" class="data row29 col0" >5.123 m, 5.123 m</td>
      <td id="T_3d8fc_row29_col1" class="data row29 col1" >False</td>
      <td id="T_3d8fc_row29_col2" class="data row29 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row29_col3" class="data row29 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row30" class="row_heading level0 row30" >lgamma</th>
      <td id="T_3d8fc_row30_col0" class="data row30 col0" >5.123 m</td>
      <td id="T_3d8fc_row30_col1" class="data row30 col1" >False</td>
      <td id="T_3d8fc_row30_col2" class="data row30 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row30_col3" class="data row30 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row31" class="row_heading level0 row31" >log</th>
      <td id="T_3d8fc_row31_col0" class="data row31 col0" >5.123 m</td>
      <td id="T_3d8fc_row31_col1" class="data row31 col1" >False</td>
      <td id="T_3d8fc_row31_col2" class="data row31 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row31_col3" class="data row31 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row32" class="row_heading level0 row32" >log10</th>
      <td id="T_3d8fc_row32_col0" class="data row32 col0" >5.123 m</td>
      <td id="T_3d8fc_row32_col1" class="data row32 col1" >False</td>
      <td id="T_3d8fc_row32_col2" class="data row32 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row32_col3" class="data row32 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row33" class="row_heading level0 row33" >log1p</th>
      <td id="T_3d8fc_row33_col0" class="data row33 col0" >5.123 m</td>
      <td id="T_3d8fc_row33_col1" class="data row33 col1" >False</td>
      <td id="T_3d8fc_row33_col2" class="data row33 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row33_col3" class="data row33 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row34" class="row_heading level0 row34" >log2</th>
      <td id="T_3d8fc_row34_col0" class="data row34 col0" >5.123 m</td>
      <td id="T_3d8fc_row34_col1" class="data row34 col1" >False</td>
      <td id="T_3d8fc_row34_col2" class="data row34 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row34_col3" class="data row34 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row35" class="row_heading level0 row35" >modf</th>
      <td id="T_3d8fc_row35_col0" class="data row35 col0" >5.123 m</td>
      <td id="T_3d8fc_row35_col1" class="data row35 col1" >False</td>
      <td id="T_3d8fc_row35_col2" class="data row35 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row35_col3" class="data row35 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row36" class="row_heading level0 row36" >perm</th>
      <td id="T_3d8fc_row36_col0" class="data row36 col0" >5.123 m</td>
      <td id="T_3d8fc_row36_col1" class="data row36 col1" >False</td>
      <td id="T_3d8fc_row36_col2" class="data row36 col2" >'Quantity' object cannot be interpreted as an integer</td>
      <td id="T_3d8fc_row36_col3" class="data row36 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row37" class="row_heading level0 row37" >pow</th>
      <td id="T_3d8fc_row37_col0" class="data row37 col0" >5.123 m, 2</td>
      <td id="T_3d8fc_row37_col1" class="data row37 col1" >False</td>
      <td id="T_3d8fc_row37_col2" class="data row37 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row37_col3" class="data row37 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row38" class="row_heading level0 row38" >prod</th>
      <td id="T_3d8fc_row38_col0" class="data row38 col0" >5.123 m, 5.123 m</td>
      <td id="T_3d8fc_row38_col1" class="data row38 col1" >True</td>
      <td id="T_3d8fc_row38_col2" class="data row38 col2" >26.245129000000002 m**2</td>
      <td id="T_3d8fc_row38_col3" class="data row38 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row39" class="row_heading level0 row39" >radians</th>
      <td id="T_3d8fc_row39_col0" class="data row39 col0" >5.123 m</td>
      <td id="T_3d8fc_row39_col1" class="data row39 col1" >False</td>
      <td id="T_3d8fc_row39_col2" class="data row39 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row39_col3" class="data row39 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row40" class="row_heading level0 row40" >remainder</th>
      <td id="T_3d8fc_row40_col0" class="data row40 col0" >5.123 m, -2 m</td>
      <td id="T_3d8fc_row40_col1" class="data row40 col1" >False</td>
      <td id="T_3d8fc_row40_col2" class="data row40 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row40_col3" class="data row40 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row41" class="row_heading level0 row41" >sin</th>
      <td id="T_3d8fc_row41_col0" class="data row41 col0" >5.123 m</td>
      <td id="T_3d8fc_row41_col1" class="data row41 col1" >False</td>
      <td id="T_3d8fc_row41_col2" class="data row41 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row41_col3" class="data row41 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row42" class="row_heading level0 row42" >sinh</th>
      <td id="T_3d8fc_row42_col0" class="data row42 col0" >5.123 m</td>
      <td id="T_3d8fc_row42_col1" class="data row42 col1" >False</td>
      <td id="T_3d8fc_row42_col2" class="data row42 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row42_col3" class="data row42 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row43" class="row_heading level0 row43" >sqrt</th>
      <td id="T_3d8fc_row43_col0" class="data row43 col0" >5.123 m</td>
      <td id="T_3d8fc_row43_col1" class="data row43 col1" >False</td>
      <td id="T_3d8fc_row43_col2" class="data row43 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row43_col3" class="data row43 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row44" class="row_heading level0 row44" >tan</th>
      <td id="T_3d8fc_row44_col0" class="data row44 col0" >5.123 m</td>
      <td id="T_3d8fc_row44_col1" class="data row44 col1" >False</td>
      <td id="T_3d8fc_row44_col2" class="data row44 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row44_col3" class="data row44 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row45" class="row_heading level0 row45" >tanh</th>
      <td id="T_3d8fc_row45_col0" class="data row45 col0" >5.123 m</td>
      <td id="T_3d8fc_row45_col1" class="data row45 col1" >False</td>
      <td id="T_3d8fc_row45_col2" class="data row45 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_3d8fc_row45_col3" class="data row45 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_3d8fc_level0_row46" class="row_heading level0 row46" >trunc</th>
      <td id="T_3d8fc_row46_col0" class="data row46 col0" >5.123 m</td>
      <td id="T_3d8fc_row46_col1" class="data row46 col1" >True</td>
      <td id="T_3d8fc_row46_col2" class="data row46 col2" >5 m</td>
      <td id="T_3d8fc_row46_col3" class="data row46 col3" >0.000000</td>
    </tr>
  </tbody>
</table>





```python
df_phymath
```




<style type="text/css">
#T_d82cd_row0_col1, #T_d82cd_row1_col1, #T_d82cd_row2_col1, #T_d82cd_row3_col1, #T_d82cd_row4_col1, #T_d82cd_row5_col1, #T_d82cd_row7_col1, #T_d82cd_row17_col1, #T_d82cd_row18_col1, #T_d82cd_row19_col1, #T_d82cd_row25_col1, #T_d82cd_row26_col1, #T_d82cd_row27_col1, #T_d82cd_row30_col1, #T_d82cd_row31_col1, #T_d82cd_row32_col1, #T_d82cd_row33_col1, #T_d82cd_row34_col1, #T_d82cd_row38_col1, #T_d82cd_row40_col1, #T_d82cd_row43_col1 {
  color: green;
}
#T_d82cd_row6_col1, #T_d82cd_row8_col1, #T_d82cd_row9_col1, #T_d82cd_row10_col1, #T_d82cd_row11_col1, #T_d82cd_row12_col1, #T_d82cd_row13_col1, #T_d82cd_row14_col1, #T_d82cd_row15_col1, #T_d82cd_row16_col1, #T_d82cd_row20_col1, #T_d82cd_row21_col1, #T_d82cd_row22_col1, #T_d82cd_row23_col1, #T_d82cd_row24_col1, #T_d82cd_row28_col1, #T_d82cd_row29_col1, #T_d82cd_row35_col1, #T_d82cd_row36_col1, #T_d82cd_row37_col1, #T_d82cd_row39_col1, #T_d82cd_row41_col1, #T_d82cd_row42_col1, #T_d82cd_row44_col1, #T_d82cd_row45_col1, #T_d82cd_row46_col1 {
  color: red;
}
</style>
<table id="T_d82cd">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d82cd_level0_col0" class="col_heading level0 col0" >Input</th>
      <th id="T_d82cd_level0_col1" class="col_heading level0 col1" >Passed</th>
      <th id="T_d82cd_level0_col2" class="col_heading level0 col2" >Returned</th>
      <th id="T_d82cd_level0_col3" class="col_heading level0 col3" >Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d82cd_level0_row0" class="row_heading level0 row0" >acos</th>
      <td id="T_d82cd_row0_col0" class="data row0 col0" >1</td>
      <td id="T_d82cd_row0_col1" class="data row0 col1" >True</td>
      <td id="T_d82cd_row0_col2" class="data row0 col2" >0.0</td>
      <td id="T_d82cd_row0_col3" class="data row0 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row1" class="row_heading level0 row1" >acosh</th>
      <td id="T_d82cd_row1_col0" class="data row1 col0" >1</td>
      <td id="T_d82cd_row1_col1" class="data row1 col1" >True</td>
      <td id="T_d82cd_row1_col2" class="data row1 col2" >0.0</td>
      <td id="T_d82cd_row1_col3" class="data row1 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row2" class="row_heading level0 row2" >asin</th>
      <td id="T_d82cd_row2_col0" class="data row2 col0" >1</td>
      <td id="T_d82cd_row2_col1" class="data row2 col1" >True</td>
      <td id="T_d82cd_row2_col2" class="data row2 col2" >1.5707963267948966</td>
      <td id="T_d82cd_row2_col3" class="data row2 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row3" class="row_heading level0 row3" >asinh</th>
      <td id="T_d82cd_row3_col0" class="data row3 col0" >1</td>
      <td id="T_d82cd_row3_col1" class="data row3 col1" >True</td>
      <td id="T_d82cd_row3_col2" class="data row3 col2" >0.8813735870195429</td>
      <td id="T_d82cd_row3_col3" class="data row3 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row4" class="row_heading level0 row4" >atan</th>
      <td id="T_d82cd_row4_col0" class="data row4 col0" >1</td>
      <td id="T_d82cd_row4_col1" class="data row4 col1" >True</td>
      <td id="T_d82cd_row4_col2" class="data row4 col2" >0.7853981633974483</td>
      <td id="T_d82cd_row4_col3" class="data row4 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row5" class="row_heading level0 row5" >atan2</th>
      <td id="T_d82cd_row5_col0" class="data row5 col0" >2.123 m, 2.123 m</td>
      <td id="T_d82cd_row5_col1" class="data row5 col1" >True</td>
      <td id="T_d82cd_row5_col2" class="data row5 col2" >0.7853981633974483</td>
      <td id="T_d82cd_row5_col3" class="data row5 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row6" class="row_heading level0 row6" >atanh</th>
      <td id="T_d82cd_row6_col0" class="data row6 col0" >1</td>
      <td id="T_d82cd_row6_col1" class="data row6 col1" >False</td>
      <td id="T_d82cd_row6_col2" class="data row6 col2" >math domain error</td>
      <td id="T_d82cd_row6_col3" class="data row6 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row7" class="row_heading level0 row7" >ceil</th>
      <td id="T_d82cd_row7_col0" class="data row7 col0" >2.123 m</td>
      <td id="T_d82cd_row7_col1" class="data row7 col1" >True</td>
      <td id="T_d82cd_row7_col2" class="data row7 col2" >3 m</td>
      <td id="T_d82cd_row7_col3" class="data row7 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row8" class="row_heading level0 row8" >coysign</th>
      <td id="T_d82cd_row8_col0" class="data row8 col0" >2.123 m, 2.123 m</td>
      <td id="T_d82cd_row8_col1" class="data row8 col1" >False</td>
      <td id="T_d82cd_row8_col2" class="data row8 col2" >module 'physipy.math' has no attribute 'coysign'</td>
      <td id="T_d82cd_row8_col3" class="data row8 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row9" class="row_heading level0 row9" >cos</th>
      <td id="T_d82cd_row9_col0" class="data row9 col0" >2.123 m</td>
      <td id="T_d82cd_row9_col1" class="data row9 col1" >False</td>
      <td id="T_d82cd_row9_col2" class="data row9 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row9_col3" class="data row9 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row10" class="row_heading level0 row10" >cosh</th>
      <td id="T_d82cd_row10_col0" class="data row10 col0" >2.123 m</td>
      <td id="T_d82cd_row10_col1" class="data row10 col1" >False</td>
      <td id="T_d82cd_row10_col2" class="data row10 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row10_col3" class="data row10 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row11" class="row_heading level0 row11" >degrees</th>
      <td id="T_d82cd_row11_col0" class="data row11 col0" >2.123 m</td>
      <td id="T_d82cd_row11_col1" class="data row11 col1" >False</td>
      <td id="T_d82cd_row11_col2" class="data row11 col2" ></td>
      <td id="T_d82cd_row11_col3" class="data row11 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row12" class="row_heading level0 row12" >dist</th>
      <td id="T_d82cd_row12_col0" class="data row12 col0" >[<Quantity : 2.123 m, symbol=m*UndefinedSymbol>, <Quantity : 2.123 m, symbol=m*UndefinedSymbol>], [<Quantity : 2.123 m, symbol=m*UndefinedSymbol>, <Quantity : 2.123 m, symbol=m*UndefinedSymbol>]</td>
      <td id="T_d82cd_row12_col1" class="data row12 col1" >False</td>
      <td id="T_d82cd_row12_col2" class="data row12 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_d82cd_row12_col3" class="data row12 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row13" class="row_heading level0 row13" >erf</th>
      <td id="T_d82cd_row13_col0" class="data row13 col0" >2.123 m</td>
      <td id="T_d82cd_row13_col1" class="data row13 col1" >False</td>
      <td id="T_d82cd_row13_col2" class="data row13 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row13_col3" class="data row13 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row14" class="row_heading level0 row14" >erfc</th>
      <td id="T_d82cd_row14_col0" class="data row14 col0" >2.123 m</td>
      <td id="T_d82cd_row14_col1" class="data row14 col1" >False</td>
      <td id="T_d82cd_row14_col2" class="data row14 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row14_col3" class="data row14 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row15" class="row_heading level0 row15" >exp</th>
      <td id="T_d82cd_row15_col0" class="data row15 col0" >2.123 m</td>
      <td id="T_d82cd_row15_col1" class="data row15 col1" >False</td>
      <td id="T_d82cd_row15_col2" class="data row15 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row15_col3" class="data row15 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row16" class="row_heading level0 row16" >expm1</th>
      <td id="T_d82cd_row16_col0" class="data row16 col0" >2.123 m</td>
      <td id="T_d82cd_row16_col1" class="data row16 col1" >False</td>
      <td id="T_d82cd_row16_col2" class="data row16 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row16_col3" class="data row16 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row17" class="row_heading level0 row17" >fabs</th>
      <td id="T_d82cd_row17_col0" class="data row17 col0" >2.123 m</td>
      <td id="T_d82cd_row17_col1" class="data row17 col1" >True</td>
      <td id="T_d82cd_row17_col2" class="data row17 col2" >2.123 m</td>
      <td id="T_d82cd_row17_col3" class="data row17 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row18" class="row_heading level0 row18" >floor</th>
      <td id="T_d82cd_row18_col0" class="data row18 col0" >2.123 m</td>
      <td id="T_d82cd_row18_col1" class="data row18 col1" >True</td>
      <td id="T_d82cd_row18_col2" class="data row18 col2" >2 m</td>
      <td id="T_d82cd_row18_col3" class="data row18 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row19" class="row_heading level0 row19" >fmod</th>
      <td id="T_d82cd_row19_col0" class="data row19 col0" >2.123 m, 2.123 m</td>
      <td id="T_d82cd_row19_col1" class="data row19 col1" >True</td>
      <td id="T_d82cd_row19_col2" class="data row19 col2" >0.0 m</td>
      <td id="T_d82cd_row19_col3" class="data row19 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row20" class="row_heading level0 row20" >fsum</th>
      <td id="T_d82cd_row20_col0" class="data row20 col0" >2.123 m, 2.123 m, 2.123 m</td>
      <td id="T_d82cd_row20_col1" class="data row20 col1" >False</td>
      <td id="T_d82cd_row20_col2" class="data row20 col2" >setting an array element with a sequence.</td>
      <td id="T_d82cd_row20_col3" class="data row20 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row21" class="row_heading level0 row21" >gamma</th>
      <td id="T_d82cd_row21_col0" class="data row21 col0" >2.123 m</td>
      <td id="T_d82cd_row21_col1" class="data row21 col1" >False</td>
      <td id="T_d82cd_row21_col2" class="data row21 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row21_col3" class="data row21 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row22" class="row_heading level0 row22" >gcd</th>
      <td id="T_d82cd_row22_col0" class="data row22 col0" >2.123 m, 2.123 m</td>
      <td id="T_d82cd_row22_col1" class="data row22 col1" >False</td>
      <td id="T_d82cd_row22_col2" class="data row22 col2" ></td>
      <td id="T_d82cd_row22_col3" class="data row22 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row23" class="row_heading level0 row23" >hypot</th>
      <td id="T_d82cd_row23_col0" class="data row23 col0" >2.123 m</td>
      <td id="T_d82cd_row23_col1" class="data row23 col1" >False</td>
      <td id="T_d82cd_row23_col2" class="data row23 col2" >Dimension error : dimension is L but should be no-dimension (length vs dimensionless).</td>
      <td id="T_d82cd_row23_col3" class="data row23 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row24" class="row_heading level0 row24" >isclose</th>
      <td id="T_d82cd_row24_col0" class="data row24 col0" >5.123 m, -2 m</td>
      <td id="T_d82cd_row24_col1" class="data row24 col1" >False</td>
      <td id="T_d82cd_row24_col2" class="data row24 col2" ></td>
      <td id="T_d82cd_row24_col3" class="data row24 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row25" class="row_heading level0 row25" >isfinite</th>
      <td id="T_d82cd_row25_col0" class="data row25 col0" >2.123 m</td>
      <td id="T_d82cd_row25_col1" class="data row25 col1" >True</td>
      <td id="T_d82cd_row25_col2" class="data row25 col2" >True</td>
      <td id="T_d82cd_row25_col3" class="data row25 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row26" class="row_heading level0 row26" >isinf</th>
      <td id="T_d82cd_row26_col0" class="data row26 col0" >2.123 m</td>
      <td id="T_d82cd_row26_col1" class="data row26 col1" >True</td>
      <td id="T_d82cd_row26_col2" class="data row26 col2" >False</td>
      <td id="T_d82cd_row26_col3" class="data row26 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row27" class="row_heading level0 row27" >isnan</th>
      <td id="T_d82cd_row27_col0" class="data row27 col0" >2.123 m</td>
      <td id="T_d82cd_row27_col1" class="data row27 col1" >True</td>
      <td id="T_d82cd_row27_col2" class="data row27 col2" >False</td>
      <td id="T_d82cd_row27_col3" class="data row27 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row28" class="row_heading level0 row28" >isqrt</th>
      <td id="T_d82cd_row28_col0" class="data row28 col0" >2.123 m</td>
      <td id="T_d82cd_row28_col1" class="data row28 col1" >False</td>
      <td id="T_d82cd_row28_col2" class="data row28 col2" ></td>
      <td id="T_d82cd_row28_col3" class="data row28 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row29" class="row_heading level0 row29" >ldexp</th>
      <td id="T_d82cd_row29_col0" class="data row29 col0" >2.123 m, 2.123 m</td>
      <td id="T_d82cd_row29_col1" class="data row29 col1" >False</td>
      <td id="T_d82cd_row29_col2" class="data row29 col2" ></td>
      <td id="T_d82cd_row29_col3" class="data row29 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row30" class="row_heading level0 row30" >lgamma</th>
      <td id="T_d82cd_row30_col0" class="data row30 col0" >1</td>
      <td id="T_d82cd_row30_col1" class="data row30 col1" >True</td>
      <td id="T_d82cd_row30_col2" class="data row30 col2" >0.0</td>
      <td id="T_d82cd_row30_col3" class="data row30 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row31" class="row_heading level0 row31" >log</th>
      <td id="T_d82cd_row31_col0" class="data row31 col0" >1</td>
      <td id="T_d82cd_row31_col1" class="data row31 col1" >True</td>
      <td id="T_d82cd_row31_col2" class="data row31 col2" >0.0</td>
      <td id="T_d82cd_row31_col3" class="data row31 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row32" class="row_heading level0 row32" >log10</th>
      <td id="T_d82cd_row32_col0" class="data row32 col0" >1</td>
      <td id="T_d82cd_row32_col1" class="data row32 col1" >True</td>
      <td id="T_d82cd_row32_col2" class="data row32 col2" >0.0</td>
      <td id="T_d82cd_row32_col3" class="data row32 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row33" class="row_heading level0 row33" >log1p</th>
      <td id="T_d82cd_row33_col0" class="data row33 col0" >1</td>
      <td id="T_d82cd_row33_col1" class="data row33 col1" >True</td>
      <td id="T_d82cd_row33_col2" class="data row33 col2" >0.6931471805599453</td>
      <td id="T_d82cd_row33_col3" class="data row33 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row34" class="row_heading level0 row34" >log2</th>
      <td id="T_d82cd_row34_col0" class="data row34 col0" >1</td>
      <td id="T_d82cd_row34_col1" class="data row34 col1" >True</td>
      <td id="T_d82cd_row34_col2" class="data row34 col2" >0.0</td>
      <td id="T_d82cd_row34_col3" class="data row34 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row35" class="row_heading level0 row35" >modf</th>
      <td id="T_d82cd_row35_col0" class="data row35 col0" >2.123 m</td>
      <td id="T_d82cd_row35_col1" class="data row35 col1" >False</td>
      <td id="T_d82cd_row35_col2" class="data row35 col2" ></td>
      <td id="T_d82cd_row35_col3" class="data row35 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row36" class="row_heading level0 row36" >perm</th>
      <td id="T_d82cd_row36_col0" class="data row36 col0" >2.123 m</td>
      <td id="T_d82cd_row36_col1" class="data row36 col1" >False</td>
      <td id="T_d82cd_row36_col2" class="data row36 col2" ></td>
      <td id="T_d82cd_row36_col3" class="data row36 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row37" class="row_heading level0 row37" >pow</th>
      <td id="T_d82cd_row37_col0" class="data row37 col0" >2.123 m, 2</td>
      <td id="T_d82cd_row37_col1" class="data row37 col1" >False</td>
      <td id="T_d82cd_row37_col2" class="data row37 col2" ></td>
      <td id="T_d82cd_row37_col3" class="data row37 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row38" class="row_heading level0 row38" >prod</th>
      <td id="T_d82cd_row38_col0" class="data row38 col0" >2.123 m, 2.123 m</td>
      <td id="T_d82cd_row38_col1" class="data row38 col1" >True</td>
      <td id="T_d82cd_row38_col2" class="data row38 col2" >4.507129000000001 m**2</td>
      <td id="T_d82cd_row38_col3" class="data row38 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row39" class="row_heading level0 row39" >radians</th>
      <td id="T_d82cd_row39_col0" class="data row39 col0" >2.123 m</td>
      <td id="T_d82cd_row39_col1" class="data row39 col1" >False</td>
      <td id="T_d82cd_row39_col2" class="data row39 col2" ></td>
      <td id="T_d82cd_row39_col3" class="data row39 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row40" class="row_heading level0 row40" >remainder</th>
      <td id="T_d82cd_row40_col0" class="data row40 col0" >2.123 m, -2 m</td>
      <td id="T_d82cd_row40_col1" class="data row40 col1" >True</td>
      <td id="T_d82cd_row40_col2" class="data row40 col2" >0.12300000000000022 m</td>
      <td id="T_d82cd_row40_col3" class="data row40 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row41" class="row_heading level0 row41" >sin</th>
      <td id="T_d82cd_row41_col0" class="data row41 col0" >2.123 m</td>
      <td id="T_d82cd_row41_col1" class="data row41 col1" >False</td>
      <td id="T_d82cd_row41_col2" class="data row41 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row41_col3" class="data row41 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row42" class="row_heading level0 row42" >sinh</th>
      <td id="T_d82cd_row42_col0" class="data row42 col0" >2.123 m</td>
      <td id="T_d82cd_row42_col1" class="data row42 col1" >False</td>
      <td id="T_d82cd_row42_col2" class="data row42 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row42_col3" class="data row42 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row43" class="row_heading level0 row43" >sqrt</th>
      <td id="T_d82cd_row43_col0" class="data row43 col0" >2.123 m</td>
      <td id="T_d82cd_row43_col1" class="data row43 col1" >True</td>
      <td id="T_d82cd_row43_col2" class="data row43 col2" >1.4570518178843195 m**0.5</td>
      <td id="T_d82cd_row43_col3" class="data row43 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row44" class="row_heading level0 row44" >tan</th>
      <td id="T_d82cd_row44_col0" class="data row44 col0" >2.123 m</td>
      <td id="T_d82cd_row44_col1" class="data row44 col1" >False</td>
      <td id="T_d82cd_row44_col2" class="data row44 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row44_col3" class="data row44 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row45" class="row_heading level0 row45" >tanh</th>
      <td id="T_d82cd_row45_col0" class="data row45 col0" >2.123 m</td>
      <td id="T_d82cd_row45_col1" class="data row45 col1" >False</td>
      <td id="T_d82cd_row45_col2" class="data row45 col2" >Dimension error : dimensions of operands are L and no-dimension, and are differents (length vs dimensionless).</td>
      <td id="T_d82cd_row45_col3" class="data row45 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_d82cd_level0_row46" class="row_heading level0 row46" >trunc</th>
      <td id="T_d82cd_row46_col0" class="data row46 col0" >2.123 m</td>
      <td id="T_d82cd_row46_col1" class="data row46 col1" >False</td>
      <td id="T_d82cd_row46_col2" class="data row46 col2" >type numpy.ndarray doesn't define __trunc__ method</td>
      <td id="T_d82cd_row46_col3" class="data row46 col3" >nan</td>
    </tr>
  </tbody>
</table>





```python

```
