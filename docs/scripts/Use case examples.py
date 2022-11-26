# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from physipy import constants
from physipy import units, m, s, K, sr
from physipy.quantity.utils import check_dimension
import matplotlib.pyplot as plt
import matplotlib.cm
# %matplotlib qt

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

# %%
integ_trapz = np.trapz(zs, x=lmbdas_).iinto(W/m**2/sr)

import physipy.integrate

integ_quad = physipy.integrate.quad(lambda lmbda:planck(lmbda, Ts_[0]), 0.1*mum, 40*mum)[0].iinto(W/m**2/sr)
print(integ_trapz, integ_quad)

# %%

# %%
