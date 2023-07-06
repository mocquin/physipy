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

# https://scipython.com/book2/chapter-6-numpy/examples/the-height-of-liquid-in-a-spherical-tank/

```python
from physipy import m, units, s, setup_matplotlib
setup_matplotlib()

import numpy as np
import matplotlib.pyplot as plt
Polynomial  = np.polynomial.Polynomial

# Radius of the spherical tank in m
R = 1.5 * m
# Flow rate out of the tank, m^3.s-1
F = 2.e-4 * m**3/s
# Total volume of the tank
V0 = 4/3 * np.pi * R**3
# Total time taken for the tank to empty
T = V0 / F

# coefficients of the quadratic and cubic terms
# of p(h), the polynomial to be solved for h
c2, c3 = np.pi * R, -np.pi / 3

N = 100
# array of N time points between 0 and T inclusive
time = np.linspace(0*s, T, N)
# create the corresponding array of heights h(t)
h = np.zeros(N)
for i, t in enumerate(time):
    c0 = F*t - V0
    p = Polynomial([c0.value, 0, c2.value, c3])
    # find the three roots to this polynomial
    roots = p.roots()
    # we want the one root for which 0 <= h <= 2R
    h[i] = roots[(0 <= roots) & (roots <= 2*R.value)][0]

h = h*m

fig, ax = plt.subplots()
ax.plot(time, h, 'o')
ax.set_xlabel("Time (" + str(ax.xaxis.get_label().get_text()) + ")")
ax.set_ylabel("Height in tank (" + str(ax.yaxis.get_label().get_text()) + ")")
plt.show()
```

# https://scipython.com/book2/chapter-6-numpy/examples/mesh-analysis-of-a-electrical-network/

```python
from physipy import m, units, s, setup_matplotlib
setup_matplotlib()
import numpy as np

ohm = units["ohm"]
volt = units["V"]

R = np.array([[50, 0, -30],
              [0, 40, -20],
              [-30, -20, 100]])*ohm
V = np.array([80, 80, 0]) * volt
I = np.linalg.inv(R) @ V
I
```


```python
import numpy as np
from physipy import constants
from physipy import units, m, s, K, sr
from physipy.quantity.utils import check_dimension
import matplotlib.pyplot as plt
import matplotlib.cm
%matplotlib qt

c = constants["c"]
h = constants["h"]
kB = constants['k']
mum = units["mum"]
W = units["W"]


@check_dimension([mum, K], W/m**2/sr/m)
def planck(lmbda, T): return 2/sr*h*c**2/lmbda**5 * 1/(np.exp(h*c/(lmbda*kB*T))-1)

N_lmbda = 300

Ts_     = np.linspace(273-40, 273+80, num=3) * K
lmbdas_ = np.linspace(0.1, 40, num=N_lmbda) * mum

lmbdas, Ts = np.meshgrid(lmbdas_, Ts_)

lums = planck(lmbdas, Ts)

fig, ax = plt.subplots()
ax.imshow(lums, interpolation='none', extent=[lmbdas.min(), lmbdas.max(), Ts.min(), Ts.max()])
ax.set_xlabel('')
ax.set_aspect('auto')
fig.tight_layout()

from mpl_toolkits.mplot3d import axes3d
fig, ax = plt.subplots(subplot_kw={"projection": "3d", 'proj_type':"ortho"})

X = ((Ts-np.min(Ts))/(np.max(Ts)-np.min(Ts)))
colors = matplotlib.pyplot.cm.magma(X)

stride = 1
ax.plot_surface(lmbdas.value, Ts.value, lums.value,
                 rstride=stride, cstride=stride,
                  facecolors=colors,
                shade=False, linewidth=0,
                 )
ax.set_ylabel('$T$')
ax.set_xlabel('$\lambda$')
ax.set_zlabel("$Lum_{\lambda}$")

xs = lmbdas_
ys = np.repeat(Ts_[0], N_lmbda)
zs = planck(xs, ys)
ax.plot(xs.value, ys.value, zs.value, color="red", linewidth=5)
```

```python
integ_trapz = np.trapz(zs, x=lmbdas_).iinto(W/m**2/sr)

import physipy.integrate

integ_quad = physipy.integrate.quad(lambda lmbda:planck(lmbda, Ts_[0]), 0.1*mum, 40*mum)[0].iinto(W/m**2/sr)
print(integ_trapz, integ_quad)
```
