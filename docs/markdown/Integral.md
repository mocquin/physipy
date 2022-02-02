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

<!-- #region tags=[] -->
# Integrals
<!-- #endregion -->

Ressources : 
 - https://medium.com/math-simplified/the-trapezoidal-rule-on-steroids-romberg-integration-4a24fba8d751
 - wikipedia on trapezoidal rule and Romberg


There are several ways to compute integrals : 
 - from a np.array, using the `.integrate()` that relies on `np.trapz`
 - use `np.trapz`
 - use `scipy.integrate.romb` or `scipy.integrate.simps`  or `scipy.integrate.trapz`
 - use `physipy.quad`, that just wraps `scipy.integrate.quad` (or dblquad or tplquad)


```python
import physipy
from physipy import m, units, s, K
import numpy as np
mm = units["mm"]
```

```python
distances = np.linspace(1, 3, num=3)*m
distances
```

## Trapezoidal rule


Split the interval in N points, including min and max, then evaluate the function you want to integrate at thoses points. Then join each point with the x-axis to create trapezoids. Those trapezoid have widths : 
$$\Delta x= \frac{b-a}{N}$$
The area of each trapezoid is just the width times the average of the function points : 
$$S_i = \frac{b-a}{N} \frac{f(x_i)+f(x_{i+1})}{2}$$
The integral is then just : 
$$I_N = \sum_{i=0}^{N-1} S_i =  \sum_{i=0}^{N-1} \frac{b-a}{N} \frac{f(x_i)+f(x_{i+1})}{2} =\frac{b-a}{N} \sum_{i=0}^{N-1} \frac{f(x_i)+f(x_{i+1})}{2} $$


Where $x_0=a$ and $x_{N-1}=b$


But notice that all points except a and b appear twice so we can factor the expression : 
$$I_N = \frac{b-a}{N} \left( \frac{f(a)+f(b)}{2} + \sum_{i=1}^{N-2}f(x_i) \right) $$


It can be shown that the error is:
$$E = \int_a^b f(x)dx - I = -\frac{(b-a)^3}{12N^2}f^{''}(\xi)$$
where $\xi$ is a number between a and b. Notice that the Error is of the opposite sign of the concavity of f between a and b (if there is no inflexion point between a and b), ie the trapezoidal rule overestimates the integral if the function is concave up.
Keep in mind that the trapezoidal error is dependent with 
$$E\propto \frac{1}{N^2}$$

```python
# computes ((1+2)/2 + (2+3)/2)
distances.integrate()
```

```python
np.trapz(distances)
```

```python
# use specific, constant spacing
dx = 1*s
# with float dx
print(np.trapz(distances, dx=1))
# with quantity dx
print(np.trapz(distances, dx=1*m))

# raw implementation
def my_trapez(f, a, b, N):
    dx = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    area = 0.5 * (y[0] + y[-1])
    area += y[1:-1].sum()
    area *= dx
    return area

my_trapez(lambda x:x, 1*m, 3*m, N=2).into(m**2)
```

This will work for integration of nd arrays. For example, computing several integrals : 

```python
# sampling
ech_t = np.linspace(1, 100)*s
# params 
ech_v = np.linspace(10, 20)*m/s
Ts, Vs = np.meshgrid(ech_t, ech_v)
D = Ts*Vs
D.integrate(axis=1, x=ech_t)
```

## Roomberg
The idea of Romberg integration is to use several time the trapezoidal rule, with various number of samples N : N, 2N, and so on. The error of the trapezoidal rule being proportionnal to $\frac{1}{N^2}$  with $\Delta x = \frac{b-a}{N}$, so the error of the trapezoidal rule is propoertionnal to $(\Delta x)^2$ : 
$$I = I_N + \alpha (\Delta x)^2$$
We can rewrite the same with 2N samples : 
$$I = I_{2N} + \alpha (\frac{\Delta x}{2})^2 = I_{2N} + \alpha (\Delta x)^2/4 $$
So equaling both : 
$$I_N + \alpha (\Delta x)^2 =  I_{2N} +  \alpha (\Delta x)^2/4 $$



So we get $\alpha$: 
$$\alpha (\Delta x)^2/4 = \frac{1}{3}(I_{2N}-I_N)$$
So the integral expression with 2N is : 
$$I = I_{2N} + \frac{1}{3}(I_{2N}-I_N) $$



The error of this method is quite less than that of the trapezoidal rule, while relying only on the trapezoidal integrals N and 2N.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def romberg(f, a, b, N):
    
    I_N = my_trapez(f, a, b, N)
    I_2N = my_trapez(f, a, b, 2*N)
    
    return I_2N + 1/3. * (I_2N - I_N)

print(my_trapez(lambda x:x, 1*m, 3*m, N=2).into(m**2))
print(romberg(lambda x:x, 1*m, 3*m, N=2).into(m**2))
```

## Simpson's 1/3 rule


http://nm.mathforcollege.com/topics/simpsons_13rd_rule.html


The trapezoidal rule apprixmates the function by a first order polynom : Simpson 1/3 rule uses a 2nd order polynom, say : 
$$P(x)=p_0 + p_1 x + p_2 x^2$$
such that 
$$P(a)=f(a)=p_0 + p_1 a + p_2 a^2$$
$$P(b)=f(b)=p_0 + p_1 b + p_2 b^2$$
$$P(m)=f(m)=p_0 + p_1 m + p_2 m^2$$
where m=(b-a)/2.



Solving for $p_0$, $p_1$, $p_2$ gives : 
$$p_0 = \frac{a^2f(b)+abf(b)-4abf(m)+abf(a)+b^2f(a)}{a^2-2ab+b^2}$$
$$p_1 = -\frac{af(a)-4af(m)+3af(b)+3bf(a)-4bf(m)+bf(b)}{a^2-2ab+b^2}$$
$$p_2 = \frac{2(f(a)-2f(m)+f(b)}{a^2-2ab+b^2}$$


On the other hand, the integral of the polynom is : 
$$ \int_{a}^b p_0 + p_1 x + p_2 x^2 dx= [xp_0 + \frac{p_1}{2}x^2 + \frac{p_2}{3}x^3]_a^b = (b-a)p_0 + \frac{p_1}{2}(b^2-a^2) + \frac{p_2}{3}(b^3-a^3)$$


Substituting $p_0$, $p_1$ and $p_2$ leads to :
$$\int_{a}^{b}f(x)dx \approx \int_{a}^{b} P(x)dx = \frac{b-a}{6}\left[f(a)+4f(\frac{a+b}{2})+f(b) \right]$$
with $h=(b-a)/2$
$$\int_{a}^{b}f(x)dx \approx \int_{a}^{b} P(x)dx = \frac{h}{3}\left[f(a)+4f(\frac{a+b}{2})+f(b) \right]$$
hence the "1/3" rule.


Another way to write the function as a polynom to integrate between a and b using a parabol, ie a second-order polynom, that has same values as the function at a, b, and middle point (a+b)/2 is using the Lagrangian interpolation:
$$\text{On [a,b]:}f(x)\approx P(x)=f(a)\frac{(x-m)(x-b)}{(a-m)(a-b)}+f(m)\frac{(x-a)(x-b)}{(m-a)(m-b)}+f(b)\frac{(x-a)(x-m)}{(b-a)(b-m)}$$


## Simpson's 3/8 rule
Similar to the quadratic approximation of 1/3 rule, but uses a cubic interpolation, which yields to : 
$$\int_{a}^{b}f(x)dx \approx \int_{a}^{b} P(x)dx = \frac{b-a}{8}\left[f(a)+3f(\frac{2a+b}{3})+3f(\frac{a+2b}{3})+f(b) \right]$$
with $h=(b-a)/3$
$$\int_{a}^{b}f(x)dx \approx \int_{a}^{b} P(x)dx = \frac{3h}{8}\left[f(a)+3f(\frac{2a+b}{3})+3f(\frac{a+2b}{3})+f(b) \right]$$
hence the "3/8" rule.

<!-- #region tags=[] -->
# Trapz for 2D integral
<!-- #endregion -->

```python
from physipy.quantity.calculus import trapz2
```

```python
#sample a 2 squared meter, in both direction with different spacing
nx = 12
ny = 30
ech_dx = np.linspace(0*m, 2*m, num=nx)
ech_dy = np.linspace(0*m, 1*m ,num=ny)
X, Y = np.meshgrid(ech_dx, ech_dy)
# make a uniform ponderation
Zs = np.ones_like(X)
print(trapz2(Zs, ech_dx, ech_dy))
```

# Scipy

```python
import scipy
```

```python
# scipy.integrate.trapz just wraps numpy's trapz
print(scipy.integrate.trapz(distances, dx=1))
print(scipy.integrate.trapz(distances, dx=1*m))
```

```python
# scipy.integrate.simps : simpson's method : approximate function's interval by polynome 
# https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Simpson
scipy.integrate.simps(distances)
scipy.integrate.simps(distances, dx=1*m)
```

```python
# scipy.integrate.romb : Romberg's method 
# https://en.wikipedia.org/wiki/Romberg%27s_method
scipy.integrate.romb(distances)
scipy.integrate.romb(distances, dx=1*m)
```

## quad

```python
def f(t):
    return t + 1*s

integ, err = physipy.quad(f, 0*s, 10*s)
integ
```

## dblquad

```python
def f(t, d):
    return (t + 1*s) * (d + 1*m)

integ, err = physipy.dblquad(f, 0*m, 10*m, 0*s, 10*s)
integ
```

## tplquad

```python
def f(t, d, deg):
    return (t + 1*s) * (d + 1*m) * (deg + 1*K)

integ, err = physipy.tplquad(f, 0*K, 10*K, 0*m, 10*m, 0*s, 10*s)
integ
```

```python

```
