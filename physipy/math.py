"""
A simple wrapped version of math module that handles Quantity
"""

from physipy import quantify, Quantity, Dimension, DimensionError

import math


implementations = {
    "one_in_same_out":[
        math.ceil,  # also delegated to __ceil__
        math.floor, # also delegated to __floor__
        math.fabs,  
        math.trunc, # also delegated to __trunc__
    ],
    "two_same_in_same_out":[
        math.fmod,
        math.remainder, # math.remainder(x, y)
    ],
    "any_bool":[
        math.isinf, 
        math.isfinite,
        math.isnan,
    ],
    "angle_or_dimless_to_dimless":[
        math.cos, 
        math.sin,
        math.tan,
        math.cosh,
        math.sinh,
        math.tanh,
    ],
    "dimless_to_radians":[
        math.acos,
        math.asin,
        math.atan,
        math.acosh,
        math.asinh,
        math.atanh,
    ],
    "not_implemented":[
        math.degrees,
        math.radians,
        math.perm, #  math.perm(n, k=None)
        math.gcd,
        math.isclose,
        math.isqrt,
        math.ldexp,
        math.pow,
    ],
    "same":[
        math.hypot, # math.hypot(*coordinates)
        math.fsum, # math.fsum(iterable)
    ],
    "prod":[
        math.prod, # math.prod(iterable, *, start=1)
    ],
    "mute":[
        math.dist,
        math.hypot,
    ],
    # dist # math.dist(p, q)
    "two_same_to_dimless":[
        math.atan2,
    ],
    "special_copysign":[
        math.copysign,
    ],
    "dimless_to_dimless":[
        math.erf,
        math.erfc,
        math.gamma,
        math.lgamma,
        math.exp,
    ],
    "any_to_same":[
        math.fsum,
    ]
    
    # math.pow(x, y)
}



def decorator_one_in_same_out(math_func):
    def decorated(x):
        x = quantify(x)
        return Quantity(math_func(x.value), x.dimension)
    return decorated

ceil  = decorator_one_in_same_out(math.ceil)
floor = decorator_one_in_same_out(math.floor)
trunc = decorator_one_in_same_out(math.trunc)
fabs  = decorator_one_in_same_out(math.fabs)


def decorator_two_same_in_same_out(math_func):
    def decorated(x, y):
        x = quantify(x)
        y = quantify(y)
        if not x.dimension == y.dimension:
            raise DimensionError(x.dimension, y.dimension)
        return Quantity(math_func(x.value, y.value), x.dimension)
    return decorated

remainder = decorator_two_same_in_same_out(math.remainder)
fmod = decorator_two_same_in_same_out(math.fmod)


def decorator_any_bool(math_func):
    def decorated(x):
        x = quantify(x)
        return math_func(x.value)
    return decorated

isinf = decorator_any_bool(math.isinf)
isfinite = decorator_any_bool(math.isfinite)
isnan = decorator_any_bool(math.isnan)


def decorator_angle_or_dimless_to_dimless(math_func):
    def decorated(x):
        x = quantify(x)
        if not (x.dimension == Dimension(None) or x.dimension == Dimension("RAD")):
            raise DimensionError(x.dimension, Dimension(None))
        return math_func(x.value)
    return decorated


cos = decorator_angle_or_dimless_to_dimless(math.cos)
sin = decorator_angle_or_dimless_to_dimless(math.sin)
tan = decorator_angle_or_dimless_to_dimless(math.tan)
cosh = decorator_angle_or_dimless_to_dimless(math.cosh)
sinh = decorator_angle_or_dimless_to_dimless(math.sinh)
tanh = decorator_angle_or_dimless_to_dimless(math.tanh)

#for f in implementations["one_in_same_out"]:
def decorator_dimless_to_dimless(math_func):
    def decorated(x):
        x = quantify(x)
        if not (x.dimension == Dimension(None)):
            raise DimensionError(x.dimension, Dimension(None))
        return math_func(x.value)
    return decorated        

acos = decorator_dimless_to_dimless(math.acos)
asin = decorator_dimless_to_dimless(math.asin)
atan = decorator_dimless_to_dimless(math.atan)
acosh = decorator_dimless_to_dimless(math.acosh)
asinh = decorator_dimless_to_dimless(math.asinh)
atanh = decorator_dimless_to_dimless(math.atanh)


def decorator_two_same_to_dimless(math_func):
    def decorated(y, x):
        y = quantify(y)
        x = quantify(x)
        return math_func(y.value, x.value)
    return decorated

atan2 = decorator_two_same_to_dimless(math.atan2)


def copysign(x, y):
    x = quantify(x)
    y = quantify(y)
    return Quantity(math.copysign(x.value, y.value), x.dimension)

def decorator_not_implemented(math_func):
    def decorated(*args):
        raise NotImplementedError
    return decorated

degrees = decorator_not_implemented(math.degrees)
radians = decorator_not_implemented(math.radians)
gcd = decorator_not_implemented(math.gcd)
isclose = decorator_not_implemented(math.isclose)
isqrt = decorator_not_implemented(math.isqrt)
ldexp = decorator_not_implemented(math.ldexp)
modf = decorator_not_implemented(math.modf)
perm = decorator_not_implemented(math.perm)
pow = decorator_not_implemented(math.pow)


def decorator_mute(math_func):
    return math_func

dist = decorator_mute(math.dist)
hypot = decorator_mute(math.hypot)
prod = decorator_mute(math.prod)

def decorator_dimless_to_dimless(math_func):
    def decorated(x):
        x = quantify(x)
        if not x.dimension == Dimension(None):
            raise DimensionError(x.dimension, Dimension(None))
        return math_func(x.value)
    return decorated

erf = decorator_dimless_to_dimless(math.erf)
erfc = decorator_dimless_to_dimless(math.erfc)
gamma = decorator_dimless_to_dimless(math.gamma)
lgamma = decorator_dimless_to_dimless(math.lgamma)
exp = decorator_dimless_to_dimless(math.exp)
log = decorator_dimless_to_dimless(math.log)
log10 = decorator_dimless_to_dimless(math.log10)
log1p = decorator_dimless_to_dimless(math.log1p)
log2 = decorator_dimless_to_dimless(math.log2)
expm1 = decorator_dimless_to_dimless(math.expm1)


def sqrt(x):
    x = quantify(x)
    return Quantity(math.sqrt(x.value), x.dimension**0.5)


def decorator_any_to_same(math_func):
    def decorated(x):
        x = quantify(x)
        return Quantity(math_func(x.value), x.dimension)
    return decorated

fsum = decorator_any_to_same(math.fsum)
#ceil = dec(math.ceil)

#math_params = {
#    "acos"     :(math.acos     , a       ),
#    "acosh"    :(math.acosh    , a       ),
#    "asin"     :(math.asin     , a       ),
#    "asinh"    :(math.asinh    , a       ),
#    "atan"     :(math.atan     , a       ),
#    "atan2"    :(math.atan2    , (a,b)   ),
#    "atanh"    :(math.atanh    , a       ),
#    #"ceil"     :(math.ceil     , a       ),
#    "coysign"  :(math.copysign , (a, b)  ),
#    #"comb":  
#    "cos"      :(math.cos      , a       ),
#    "cosh"     :(math.cosh     , a       ),
#    "degrees"  :(math.degrees  , a       ),
#    #"dist"     :(dist, ),
#    "erf"      :(math.erf      , a       ),
#    "erfc"     :(math.erfc     , a       ),
#    "exp"      :(math.exp      , a       ),
#    "expm1"    :(math.expm1    , a       ),
#    #"fabs"     :(math.fabs     , a       ),
#    "fmod"     :(math.fmod     , (a, b)  ),
#    "fsum"     :(math.fsum     , [a, a, a]),
#    "gamma"    :(math.gamma    , a       ),
#    "gcd"      :(math.gcd      , (a, a)  ),
#    "hypot"    :(math.hypot    , a       ),
#    "isclose"  :(math.isclose  , (a, b)  ),
#    "isfinite" :(math.isfinite , a       ),
#    "isinf"    :(math.isinf    , a       ), 
#    "isnan"    :(math.isnan    , a       ), 
#    "isqrt"    :(math.isqrt    , a       ),
#    "ldexp"    :(math.ldexp    , (a, a)  ), 
#    "lgamma"   :(math.lgamma   , a       ),
#    "log"      :(math.log      , a       ),
#    "log10"    :(math.log10    , a       ),
#    "log1p"    :(math.log1p    , a       ),
#    "log2"     :(math.log2     , a       ),
#    "modf"     :(math.modf     , a       ),
#    "perm"     :(math.perm     , a       ),
#    "math_pow" :(math.math_pow , (a, 2)  ),
#    "prod"     :(math.prod     , [a, a]  ),
#    "radians"  :(math.radians  , a       ),
#    "remainder":(math.remainder, (a, b)  ),
#    "sin"      :(math.sin      , a       ), 
#    "sinh"     :(math.sinh     , a       ), 
#    "sqrt"     :(math.sqrt     , a       ),
#    "tan"      :(math.tan      , a       ), 
#    "tanh"     :(math.tanh     , a       ),
#    "trunc"    :(math.trunc    , a       ),
#}