## Use of `scipy` in `physipy`
`scipy` is used in `physipy` for 2 reasons : 

1. To define the values of the physical constants available in `physipy.constants`
2. To provide wrapped versions of usefull `scipy` functions and make them unit-aware, available in `physipy.calculus`

It could be discussed as constants' values could be hardcoded, and wrapped functions could be defined by the user on the go. This way `scipy` would not be a dependency of `physipy`

### Constants
See the [constant section of the quickstart](./../quickstart.md).

### Wrapped functions
Some functions are regularly used in the physics/engineering world, hence we provide some functions that wrapped the units around the underlying `scipy` functions. Those functions are : 

 - quad
 - dblquad
 - tplquad
 - solve_ivp
 - root
 - brentq

#### Integrals `quad`

Those functions can be used to compute integral of functions from `a` to `b`:


```python
from physipy import s
from physipy.calculus import quad

def toto(t):
    return 2*s + t

solution, abserr = quad(toto, 0*s, 5*s)
print(solution)
```

    22.5 s**2
    

You can compute integrals of 2D and 3D functions using `dblquad` and `tplquad` respectively.

#### Initial Value Problem of ODE system
Solve an initial value problem for a system of ODEs. This function numerically integrates a system of ordinary differential equations given an initial value:
        dy / dt = f(t, y)
        y(t0) = y0



```python
from physipy.calculus import solve_ivp
from physipy import s, units

# voltage unit
V = units['V']

def exponential_decay(t, y): return -0.5 * y

sol = solve_ivp(exponential_decay, [0, 10]*s, [2, 4, 8]*V)
print(sol.t)    # time samples
print(sol.y)    # voltage response
```

    [ 0.          0.11487653  1.26364188  3.06061781  4.81611105  6.57445806
      8.33328988 10.        ] s
    [<Quantity : [2.         1.88836035 1.06327177 0.43319312 0.18017253 0.07483045
     0.03107158 0.01350781] kg*m**2/(A*s**3)>, <Quantity : [4.         3.7767207  2.12654355 0.86638624 0.36034507 0.14966091
     0.06214316 0.02701561] kg*m**2/(A*s**3)>, <Quantity : [8.         7.5534414  4.25308709 1.73277247 0.72069014 0.29932181
     0.12428631 0.05403123] kg*m**2/(A*s**3)>]
    

#### Root solver `root`

A wrapper of `scipy.optimize.root`:


```python
from physipy import s
from physipy.calculus import root

def toto(t):
    return -10*s + t

# Find the root for toto(t) = 0*s
print(root(toto, 0*s))

```

    10.0 s
    

The wrapped function signature is the same as the original's one, so additionnal args and kwargs still works :


```python
def tata(t, p):
    return -10*s*p + t

# Find the root for tata(t, 0.5) = 0*s
print(root(tata, 0*s, args=(0.5,)))
```

    5.0 s
    

#### Quadratic Brent method `brentq`
Find a root of a function in a bracketing interval using Brent's method, a wrapper of `scipy.optimize.brentq`:


```python
from physipy.calculus import brentq
```


```python
# find the solutition for toto(t) = 0*s for t in [-10, 10]*s
print(brentq(toto, -10*s, 10*s))

print(brentq(tata, -10*s, 10*s, args=(0.5,)))
```

    10.0 s
    5.0 s
    

### Note
If you want support for other scipy functions, you can either define it yourself (use the functions above as examples), or open an issue on the github page.
