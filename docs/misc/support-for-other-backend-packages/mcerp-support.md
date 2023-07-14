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

TODO : 
 - how to make UV attributes like std, skew, accessible ? using q.value.skew ? 
 - multiplication precedence : `x = mcerp.Normal(1, 2)*s` doesn't work because mcerp can't cast quantity object to UV


# mcerp
See :
 - github : https://github.com/tisimst/mcerp
 - online doc : https://pythonhosted.org/mcerp/


# Introduction on mcerp


UV : uncertain Variable


liste of distributions : https://pythonhosted.org/mcerp/distribution_constructors.html

```python
import mcerp
mcerp.npts = 12_000
```

```python
x = mcerp.Normal(1, 3, tag="toto")
y = mcerp.Normal(2, 5)
print(x)
print(y)
print(x.mean)
print(x.var)
print(x.kurt)
print(x.skew)
print(mcerp.npts)
print(x.tag)
```

```python
x.describe()
y.describe()
x.plot()
y.plot()
print(x.tag)
```

```python
z = x + y
```

```python
z.describe()
z.plot()
```

```python
print(2*z)
print(z*2)
print(2**z)
print(z**2)
print(2/z)
print(z/2)
#print(2//z)
#print(z//2)
```

# Support with physipy

```python
import physipy 
from physipy import m, cd, s, Quantity, Dimension
```

## Construction

```python
x = Quantity(mcerp.Normal(1, 2), Dimension("L"))
x
```

```python
# using multiplication
# x = mcerp.Normal(1, 2)*s : NotUpcast: <class 'physipy.quantity.quantity.Quantity'> cannot be converted to a number with uncertainty
x = m *mcerp.Normal(1, 2)
x
```

## Basic Operation

```python
# with scalar
scalar = 2*m
print(x+scalar)
print(x-scalar)
print(x*scalar)
print(x/scalar)
# print(x**s) : NotUpcast: <class 'physipy.quantity.quantity.Quantity'> cannot be converted to a number with uncertainty

print(scalar+x)
print(scalar-x)
print(scalar*x)
print(scalar/x)
# print(x**s) : NotUpcast: <class 'physipy.quantity.quantity.Quantity'> cannot be converted to a number with uncertainty
```


```python
# with other uv
x = Quantity(mcerp.Normal(1, 2), Dimension("L"))
y = Quantity(mcerp.Normal(1, 2), Dimension("L"))

print(x+y)
print(x-y)
print(x*y)
print(x/y)
# print(x**y) : NotUpcast: <class 'physipy.quantity.quantity.Quantity'> cannot be converted to a number with uncertainty

print(y+x)
print(y-x)
print(y*x)
print(y/x)
# print(y**x) : NotUpcast: <class 'physipy.quantity.quantity.Quantity'> cannot be converted to a number with uncertainty
```

# Functionnalities
Use `q.value.` to acces mcerp attributes

```python
x.value.describe()
x.value.skew
print(x.value.stats)
```

# Examples

```python
from mcerp import N, U, Gamma, Beta, Exp, H
```

```python
x1 = m * N(24, 1)
x2 = m * N(37, 4)
x3 = s * Exp(2)
```

```python
# x1.mean
print("x1.mean")
print(x1.mean) # falls back on UV value, hence drops the unit
# x1.mean() : fails because relies on numpy.mean

# x1.var
print("x1.var")
print(x1.var) # falls back on UV value, hence drops the nit

# x1.skew
print("x1.skew")
print(x1.skew) # falls back on UV value, hence drops the nit

# x1.kurt
print("x1.kurt")
print(x1.kurt)

# x1.stats
print("x1.stats")
print(x1.stats)

# Z = (x1*x2**2)/(15*(1.5 + x3))
Z = (x1*x2**2)/(15*(1.5*s + x3))
print(Z)

# Z.describe()
Z.describe()
```

# Distributions

```python
mu = 1
sigma = 0.1
x = s * mcerp.Normal(mu, sigma)
```

```python
# y = mcerp.Normal(mu*s, sigma*s) # fails because "assert sigma > 0, "
```

```python
import scipy.stats as ss
x = s * mcerp.uv(ss.norm(loc=10, scale=1))
x
```

```python
distrib = ss.norm(loc=10*s, scale=1*s)
distrib.mean() # unit is stripped, not mcerp's fault
```

```python
y = mcerp.uv(distrib)
y
```

# Plotting

```python
# x1.plot() : fails because uses Quantity.plot, which doesn't handle mcerp
```

```python
x1.value.plot()
```

```python
# fails because of matploltib version on normed 
x1.value.plot(hist=True)
```

```python
rvs1 = m*N(5, 10)
rvs2 = m*N(5, 10) + m*N(0, 0.2)
rvs3 = m*N(8, 10) + m*N(0, 0.2)
```

```python
from scipy.stats import ttest_rel
tstat, pval = ttest_rel(rvs1._mcpts, rvs2._mcpts)
pval
```

```python
rvs1<rvs2
```

```python
tstat, pval = ttest_rel(rvs1._mcpts, rvs3._mcpts)
pval
```

```python
float(tstat)
```

```python
rvs1<rvs3
```

```python
x = N(0, 1)
y = N(0, 10)
x<y
```

```python
x>y
```

```python
x==y
```

```python
x1==x1
```

```python
n1 = s*N(0, 1)
n2 = s*N(0, 1)
n1==n2
```

```python
print(Z*Z)
print(Z**2)
Z*Z == Z**2
```

```python
h = s*H(50, 5, 10)
h==4
```

```python
h<=3*s
```

```python
n = s*N(0, 1)
n==0*s
```

```python
n==0.5*s
```

```python
n==1.2345*s
```

```python

```
