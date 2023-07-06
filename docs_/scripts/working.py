# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transpose attribute error

# %%
import numpy as np
print(np.__version__)

# %%
import physipy
from physipy import m
q_test = 5*m
print(type(q_test.value))

# %%
a = np.array(5, dtype=np.float64)*m

# %%
a.T

# %%
np.array(5).T

# %%
q_test.T

# %%
q_test.value.T

# %% [markdown]
# # 2020-10-20 hasattr and getattr

# %% [markdown]
# [The docs on hasattr](https://docs.python.org/3/library/functions.html#hasattr) states : 
# ```
# hasattr(object, name)¶
#
#     The arguments are an object and a string. The result is True if the string is the name of one of the object’s attributes, False if not. (This is implemented by calling getattr(object, name) and seeing whether it raises an AttributeError or not.)
# ```
# The "problem" is that the `__getattr__` method is called only after getattr has failed : see [here](https://stackoverflow.com/questions/1944625/what-is-the-relationship-between-getattr-and-getattr).
# This could be solved with monkey-patching  by declaring `__len__` method after the value is defined depending on its type (scalar or iterable).
# See also : https://stackoverflow.com/questions/52940409/detect-if-a-getattribute-call-was-due-to-hasattr

# %%
# With float
f = 5
print(hasattr(f, "__len__"))

from physipy import s
# With q
q = 5*s
print(hasattr(q, "__len__"))

# based on getattr
#print(getattr(f, "__len__")) #--> raises an AttributeError
print(getattr(q, "__len__")) #--> doesn't !
# hence the difference with
# q.__getattr__("__len__") #--> raises an AttributeError

# %%
print(hasattr(f, "__iter__"))
print(hasattr(q, "__iter__"))

# %% [markdown]
# How does pint do on this ?

# %%
import pint

# %%
ureg = pint.UnitRegistry()
qp = 5*ureg.s
print(qp)

# %%
print(hasattr(qp, "__len__")) # same result as physipy
print(getattr(qp, "__len__")) # same
# iteration raises TypeError: 'int' object is not iterable
#for i in qp:
#    print(i)

# %% [markdown]
# --> Pint has same problems

# %% [markdown]
# How does forallpeople do ? 

# %%
import forallpeople
import forallpeople as si

# %%
qfap = 5*si.m
print(qfap)

# %%
print(hasattr(qfap, "__len__"))
#print(getattr(qfap, "__len__"))

# %% [markdown]
# But does it work with arrays ?

# %%
import numpy as np
qarrfap = np.arange(3)*si.m
print(qarrfap)
print(qarrfap*2)
print(type(qarrfap))
print(type(qfap))

# %% [markdown]
# Works with arrays, but arrays are subclassed from numpy, while scalars are Physical.

# %% [markdown]
# ## 2020-10-20 canary method

# %% [markdown]
# Using `try:getattr(self, item)` in `__getattr__` was a mistake because `__getattr__` is called after it failed to find item in object's attrs. Hence created an infinite loop, and doesn't return an AttributeError, hence `hasattr` would never return AttributeError, hence the ipython_canary_method raised.
#  - https://stackoverflow.com/questions/49822323/implications-of-the-ipython-canary-method-and-what-happens-if-it-exists
#  - https://stackoverflow.com/questions/56068348/what-is-ipython-canary-method-should-not-exist

# %% [markdown]
# ## 2020-10-21 Handcalcs support

# %%
from handcalcs import render
import handcalcs
print(handcalcs.__file__)
from physipy import m, s, K
import physipy

# %%
from scipy.integrate import quad
def f(x): 
    return 2*x + 3


# %%
# %%render
a = s

# %% [markdown]
# This happens because the equality test compares a quantity with a string.
# This is corrected by adding 
# ```python
#     def __eq__(self,y):
#         try:
#             y = quantify(y)
#             if self.dimension == y.dimension:
#                 return self.value == y.value # comparing arrays returns array of bool
#             else:
#                 return False
#         except Exception as e:
#             return False
# ```
# see commit c11cc5c

# %%
# %%render 
b = 2*s

# %% [markdown]
# This happens because Quantity doesn't have any latex repr method.

# %% [markdown]
# Changing physipy :

# %% [markdown]
# https://stackoverflow.com/questions/31291608/effect-of-using-sys-path-insert0-path-and-sys-pathappend-when-loading-modul

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
#print(sys.path)
import physipy
print(physipy.__file__)
from handcalcs import render
import handcalcs
print(handcalcs.__file__)
from physipy import m, s, K


# %%
# %%render
a = s

# %%
# %%render 
b = 2*s

# %% [markdown]
# Need to add a line in hancalcs see https://github.com/connorferster/handcalcs/issues/58

# %% [markdown]
# ## Numpy fft support
# The functions in the module are https://numpy.org/doc/stable/reference/routines.fft.html.
#
# Standard FFTs
#  - fft(a[, n, axis, norm]) : Compute the one-dimensional discrete Fourier Transform.
#  - ifft(a[, n, axis, norm]) :  Compute the one-dimensional inverse discrete Fourier Transform.
#  - fft2(a[, s, axes, norm]) Compute the 2-dimensional discrete Fourier Transform
#  - ifft2(a[, s, axes, norm]) Compute the 2-dimensional inverse discrete Fourier Transform.
#  - fftn(a[, s, axes, norm])	 Compute the N-dimensional discrete Fourier Transform.
#  - ifftn(a[, s, axes, norm]) Compute the N-dimensional inverse discrete Fourier Transform.
#  
# Real FFTs
#  - rfft(a[, n, axis, norm]) Compute the one-dimensional discrete Fourier Transform for real input.
#  - irfft(a[, n, axis, norm]) Compute the inverse of the n-point DFT for real input.
#  - rfft2(a[, s, axes, norm]) Compute the 2-dimensional FFT of a real array.
#  - irfft2(a[, s, axes, norm]) Compute the 2-dimensional inverse FFT of a real array.
#  - rfftn(a[, s, axes, norm]) Compute the N-dimensional discrete Fourier Transform for real input.
#  - irfftn(a[, s, axes, norm]) Compute the inverse of the N-dimensional FFT of real input.
#  
# Hermitian FFTs
#  - hfft(a[, n, axis, norm]) Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum.
#  - ihfft(a[, n, axis, norm]) Compute the inverse FFT of a signal that has Hermitian symmetry.
#  
# Helper routines
#  - fftfreq(n[, d]) Return the Discrete Fourier Transform sample frequencies.
#  - rfftfreq(n[, d]) Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft).
#  - fftshift(x[, axes]) Shift the zero-frequency component to the center of the spectrum.
#  - ifftshift(x[, axes]) The inverse of fftshift.
#  
# Remember that a is a vector of values, that can be complex.
# See 
# https://en.wikipedia.org/w/index.php?title=Fourier_transform&action=edit&section=15
# Remember that the DFT is : 
# $$X_k = \sum_{n=0}^{N-1}x_n e^{-\frac{i2\pi}{N}kn}$$
# where the $x_n$ can have any dimension, and the argument of the exp must be dimensionless, $n$, $N$, and so $k$ are dimensionless integers. And so $X_k$ has same dimension as $x_n$.

# %% [markdown]
# Let's try : 

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
#print(sys.path)
import physipy
print(physipy.__file__)

from physipy import m, s, K
import numpy as np

# %%
np.fft.fft(np.arange(10)*K)

# %% [markdown]
# It seems to rely on the \_\_array_function\__ --> that is good news because it means we can hook our implementation.

# %% [markdown]
# Working on np fft branch.
# In signature "a, n, axis, norm", only a can be a quantity and must be handled by physipy.
#

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
#print(sys.path)
import physipy
print(physipy.__file__)

from physipy import m, s, K
import numpy as np

# %%
np.fft.fft(np.arange(10)*K)

# %% [markdown]
# The same process must be applied on all fft/ifft.
# Only the helpers routines need additional care on dimension BUT, [since the computation with the unit is simply a division](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq), which is already handled, there is no need for a hook-wrap.

# %%
print(np.fft.fftfreq(10))
print(np.fft.fftfreq(10, 1))
print(np.fft.fftfreq(10, 1*s))
print(np.fft.rfftfreq(10, 1))
print(np.fft.rfftfreq(10, 1*s))

# %% [markdown]
# Finaly

# %%
numpy.fft.fftshift(np.arange(10))
numpy.fft.fftshift(np.arange(10)*s)

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
#print(sys.path)
import physipy
print(physipy.__file__)

from physipy import m, s, K
import numpy as np

# %%
print(numpy.fft.fftshift(np.arange(10)))
print(numpy.fft.fftshift(np.arange(10)*s))

# %% [markdown]
# Things to test in unittests : 
#  - scalars/arrays
#  - complex values
#  - Dimensionless objects
# Should consider implementing scipy's functions ? 
# https://docs.scipy.org/doc/scipy/reference/fft.html#module-scipy.fft

# %% [markdown]
# ## 2020-10-24 Matplotlib spectrum plot
#
#  - https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/spectrum_demo.html
#  - https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.magnitude_spectrum.html
#  - https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.psd.html#matplotlib.pyplot.psd
#  - https://matplotlib.org/3.3.1/gallery/lines_bars_and_markers/psd_demo.html#sphx-glr-gallery-lines-bars-and-markers-psd-demo-py
#
# Matplolib defines functions to easy plot spectrum of signals
# Is physipy plug-and-play with these functions ?

# %%
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)

dt = 0.01  # sampling interval
Fs = 1 / dt  # sampling frequency
t = np.arange(0, 10, dt)

# generate noise:
nse = np.random.randn(len(t))
r = np.exp(-t / 0.05)
cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)]

s = 0.1 * np.sin(4 * np.pi * t) + cnse  # the signal

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

# plot time signal:
axes[0, 0].set_title("Signal")
axes[0, 0].plot(t, s, color='C0')
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")

# plot different spectrum types:
axes[1, 0].set_title("Magnitude Spectrum")
axes[1, 0].magnitude_spectrum(s)#, Fs=Fs, color='C1')

axes[1, 1].set_title("Log. Magnitude Spectrum")
axes[1, 1].magnitude_spectrum(s)#, Fs=Fs, scale='dB', color='C1')

axes[2, 0].set_title("Phase Spectrum ")
axes[2, 0].phase_spectrum(s)#, Fs=Fs, color='C2')

axes[2, 1].set_title("Angle Spectrum")
axes[2, 1].angle_spectrum(s)#, Fs=Fs, color='C2')

axes[0, 1].remove()  # don't display empty ax

fig.tight_layout()
plt.show()

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s

np.random.seed(0)

dt = 0.01*physipy.s  # sampling interval
Fs = 1 / dt  # sampling frequency
t = np.arange(0, 10, dt/s)*s

# %%
# generate noise:
nse = np.random.randn(len(t))
r = np.exp(-t / (0.05*s))
cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)]


# %%
signal = 0.1*s * np.sin((4/s) * np.pi * t) + cnse  # the signal

# %%
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

# plot time signal:
#axes[0, 0].set_title("Signal")
axes[0, 0].plot(t, signal, color='C0')
#axes[0, 0].set_xlabel("Time")
#axes[0, 0].set_ylabel("Amplitude")

# plot different spectrum types:
#axes[1, 0].set_title("Magnitude Spectrum")
res = axes[1, 0].magnitude_spectrum(signal, Fs=Fs, Fc=0*1/s, color='C1', pad_to=2*len(t))

#axes[1, 1].set_title("Log. Magnitude Spectrum")
axes[1, 1].magnitude_spectrum(signal, Fs=Fs, Fc=0/s, scale='dB', color='C1')

#axes[2, 0].set_title("Phase Spectrum ")
axes[2, 0].phase_spectrum(signal, Fs=Fs, Fc=0/s, color='C2')

#axes[2, 1].set_title("Angle Spectrum")
axes[2, 1].angle_spectrum(signal, Fs=Fs, Fc=0/s, color='C2')

axes[0, 1].remove()  # don't display empty ax

fig.tight_layout()
plt.show()

# %% [markdown]
# Problem : when calling with arg Fs, the freqs for xaxis are computed automaticaly with the units (great) and added to Fc which defaults to integer 0 (bad)
# To avoid, need to add Fc=0/s to use Fs
# For the same reason, axes[1, 0].magnitude_spectrum(signal, Fc=0*1/s, color='C1')will raise DimensionError
# The pad argument is a number of samples, so dimensionless.
# Todo : test the values returned by the fucntions (here just the plot)

# %% [markdown]
# To tests:
#
# no args
# axes[1, 0].magnitude_spectrum(s)
# axes[2, 0].phase_spectrum(s)
# axes[2, 1].angle_spectrum(s)
#
# with Fs and Fc args
# axes[1, 0].magnitude_spectrum(signal, Fs=Fs, Fc=0/s, color='C1')
# axes[1, 1].magnitude_spectrum(signal, Fs=Fs, Fc=0/s, scale='dB', color='C1')
# axes[2, 0].phase_spectrum(signal, Fs=Fs, Fc=0/s, color='C2')
# axes[2, 1].angle_spectrum(signal, Fs=Fs, Fc=0/s, color='C2')
#

# %% [markdown]
# ## 2020-10-25 np.arange implement arange like linspace
# about pint
#  - https://github.com/hgrecco/pint/issues/484
#  - https://github.com/numpy/numpy/issues/12379
#  
# No clean way to override numpy. propose to use either : 
# np.arange(0, 10, 0.1)*s
# or use the helper function 

# %%

# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s
from physipy import quantify, Quantity

#def qarange(start_or_stop, stop=None, step=None, *args, **kwargs):
#    """Wrapper around np.arange"""
#    # start_or_stop param
#    final_start_or_stop = quantify(start_or_stop)
#    in_dim = final_start_or_stop.dimension
#    
#    # stop param
#    if stop is None:
#        final_stop = Quantity(1, in_dim)
#    else:
#        final_stop = quantify(stop)
#    if not final_stop.dimension == final_start_or_stop.dimension:
#        raise DimensionError(final_start_or_stop.dimension, final_stop.dimension)
#    
#    # step param
#    if step is None:
#        final_step = Quantity(0.1, in_dim)
#    else:
#        final_step = quantify(step)
#    if not final_step.dimension == final_start_or_stop.dimension:
#        raise DimensionError(final_start_or_stop.dimension, final_step.dimension)
#
#    # final call
#    val = np.arange(final_start_or_stop.value, final_stop.value, final_step.value, *args, **kwargs)
#    res = Quantity(val, in_dim)
#    return res

from physipy.quantity.utils import qarange

print("toto")
a = qarange(0*s, step=0.1*s)
b = np.arange(0, step=0.1)*s
print(a)
print(b)
print(np.all(a==b))
#print(qarange(0*s, step=0.1*s) == np.arange(0, step=0.1)*s)



# %% [markdown]
# ## 2020-10-26 plt.psd

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.01*s
t = np.arange(0, 10, dt/s)*s

# %%
nse = np.random.randn(len(t))
r = np.exp(-t / (0.05*s))

# %%
cnse = np.convolve(nse, r) * dt/s
cnse = cnse[:len(t)]
signal = 0.1 * np.sin(2/s * np.pi * t) + cnse

# %%
plt.subplot(211)
plt.plot(t, signal)
plt.subplot(212)
plt.psd(signal)#, 512, 1 / dt)
plt.show()

# %%
plt.subplot(211)
plt.plot(t, signal)
plt.subplot(212)
plt.psd(signal, 512)#, 1 / dt)
plt.show()

# %%
plt.subplot(211)
plt.plot(t, signal)
plt.subplot(212)
plt.psd(signal, 512, 1 /dt/s)
plt.show()

# %% [markdown]
# The problem : the spectrum is plotted with 10*log(pxx) where pxx has dimension, but log doesn't accept dimension

# %%
import matplotlib.mlab as mlab

Fs = 1 / dt
pxx, freqs = mlab.psd(x=signal, Fs=Fs)#, detrend=detrend,
                              #window=window, noverlap=noverlap, pad_to=pad_to,
                              #sides=sides, scale_by_freq=scale_by_freq)
print(pxx.dimension)
print(freqs.dimension)


# %% [markdown]
# ## 2020-10-26 implement np.convolve

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s

# %%
arr = np.arange(10)*s
str(np.convolve(arr, arr))
np.convolve(arr, np.ones(10))

# %% [markdown]
# ## 2020-10-26 implement numpy.blackman, numpy.hamming, numpy.bartlett
# these are filter that do not require units

# %% [markdown]
# ## 2020-10-26 pickle
#

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s

# %%
import pickle

# %%
picklefile = open('laptop1', 'wb')

# %%
q_m = 5*physipy.m
pickle.dump(q_m, picklefile)

# %%
picklefile.close()

# %%
picklefile = open('laptop1', 'rb')

# %%
laptop1 = pickle.load(picklefile)

# %%
print(laptop1)

# %% [markdown]
# ## 2020-10-27 array to latex

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s

# %%
np.arange(5)

# %%
print(np.arange(5))

# %%
print(str(np.arange(5)))

# %%
np.arange(5)

# %%
import pint

# %%
ureg = pint.UnitRegistry()
arr = np.arange(3)*ureg.m

# %%
arr

# %%
print(arr)

# %%
print(str(arr))

# %%
arr = np.arange(6).reshape(3,2)*ureg.m
arr

# %%
5*ureg.m

# %% [markdown]
# ## 2020-10-28 Dimension latex repr

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s

from physipy import Dimension
from IPython.display import display, Latex

# %%
a = Dimension(None)
b = Dimension("T")
c = Dimension({"T":3})
d = Dimension({"T":-4})
e = Dimension({"T":2, "L":-3})

# %%
print(a)
print(b)
print(c)
print(d)
print(e)
print()

# %%
print(str(a))
print(str(b))
print(str(c))
print(str(d))
print(str(e))
print()

# %%
print(repr(a))
print(repr(b))
print(repr(c))
print(repr(d))
print(repr(e))
print()

# %%
print(a.str_SI_unit()) # empty string
print(b.str_SI_unit())
print(c.str_SI_unit())
print(d.str_SI_unit())
print(e.str_SI_unit())
print()

# %%
print(a._repr_latex_())
print(b._repr_latex_())
print(c._repr_latex_())
print(d._repr_latex_())
print(e._repr_latex_())
print()

# %%
print(a.latex_SI_unit())
print(b.latex_SI_unit())
print(c.latex_SI_unit())
print(d.latex_SI_unit())
print(e.latex_SI_unit())

# %%
display(a) # calls _repr_latex 
display(b)
display(c)
display(d)
display(e)

# %%
display(Latex(a.latex_SI_unit())) # display calls _repr_latex_
display(Latex(b.latex_SI_unit()))
display(Latex(c.latex_SI_unit()))
display(Latex(d.latex_SI_unit()))
display(Latex(e.latex_SI_unit()))
display(a.latex_SI_unit()) # display calls _repr_latex_
display(b.latex_SI_unit())
display(c.latex_SI_unit())
display(d.latex_SI_unit())
display(e.latex_SI_unit())


# %% [markdown]
#  `display(object)` : get hooked on `_repr_latex_` and render as latex  
#  `display(Latex(a.other_repr_latex_()))` will also be rendered as latex  
#  `display("$1+1$")` will render the "$1+1$" string as a string  

# %%
a = Dimension({"T":2, "L":-3})
print(a)
print(str(a))
print(repr(a))
print(a.str_SI_unit()) # empty string
print(a._repr_latex_())
print(a.latex_SI_unit())
display(a) # calls _repr_latex 
display(Latex(a.latex_SI_unit())) # display calls _repr_latex_
display(a.latex_SI_unit()) # display calls _repr_latex_

print()
a.DEFAULT_REPR_LATEX = "SI_unit"
print(a)
print(str(a))
print(repr(a))
print(a.str_SI_unit()) # empty string
print(a._repr_latex_())
print(a.latex_SI_unit())
display(a) # calls _repr_latex 
display(Latex(a.latex_SI_unit())) # display calls _repr_latex_
display(a.latex_SI_unit()) # display calls _repr_latex_

b = Dimension({"T":1, "theta":2})
display(b)

# %%
b

# %% [markdown]
# There is a difference between class attribute and instance attribute : 

# %%
a.__dict__

# %%
a.__class__.__dict__

# %%

# %%
import sympy.printing.latex as latex
from sympy.parsing.sympy_parser import parse_expr

# %%
parsed = parse_expr("T**2/L**3")
print(parsed)
power_dic = parsed.as_powers_dict()
print(power_dic)
lat = "$"+latex(parsed)+"$"
display(lat)

# %% [markdown]
# ## 2020-10-28 quantity are repr latex with to many digits for lisibility in handcalcs

# %%
a = 1.23456789101112233445
print(type(a))
print(a)
print(str(a))
display(a)
a

# %%
import sympy as sp

sp.init_printing()
a

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np
import physipy
from physipy import s
from handcalcs import render
import forallpeople as si

from physipy import Dimension
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp

sp.init_printing()

# %%
ureg = pint.UnitRegistry()
b = a*ureg.m
b

# %%
f = 4.123456*si.m

# %%
# %%render
b = 1.122334456778 * ureg.m

# %%
5.12345678456789*s

# %%

# %%
print(f"{4.1234*s:.2f}")
print(f"{4.1234*s**2:.2f}")
b = 4.12345*s**2
b.favunit = physipy.units["ms"]
print(f"{b:.2f}")
c = 0.2345*s
c.symbol = "0.2345s"
b.favunit = c
print(b)
print(f"{b:.2f}")



# %% [markdown]
# ## 2020-10-29
#

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp


mm = units['mm']

sp.init_printing()

# %%
a = 3*s
a

# %%
b = 3.12345678*s
b

# %%
b.DIGITS = 5

# %%
b

# %%
# %%render
a = 1.12345*s

# %%
c = 3*s**2
c

# %%
d = 3.1223*s**2/m**3
d

# %%
mm

# %%
d = 0.01*mm
d

# %%
d.EXP_THRESH = 5

# %%
d

# %% [markdown]
# ## 2020-10-30 pick smart favunit

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp


mm = units['mm']
km = units["km"]
print(mm)


# %%
import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# %%

def list_of_Q_to_Q_array(Q_list):
    """Convert list of Quantity's object to a Quantity with array value.
    All Quantity must have the same dimension."""
    first = quantify(Q_list[0])
    dim = first.dimension
    val_list = []
    for q in Q_list:
        q = quantify(q)
        if q.dimension == dim:
            val_list.append(q.value)
        else:
            raise ValueError
    return Quantity(np.array(val_list), dim)


#np.array([0.001, 1])*m == list_of_Q_to_Q_array([mm, m])




# %%
my_list = [mm, m, km]
ar = list_of_Q_to_Q_array(my_list)
print(ar)
#units_array = physipy.quantity.utils.array_to_Q_array(ar)
#units_array

# %%
a = physipy.quantity.utils.array_to_Q_array(ar)
print(a)
a

# %% [markdown]
# ## Dimension repr latex tests

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp


mm = units['mm']
km = units["km"]
print(mm)


# %% [markdown]
# ## 2020-10-31 array rerp

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp


mm = units['mm']
km = units["km"]
cm = units["cm"]
print(mm)


arra = np.arange(10)*m
arrb = np.arange(10)*mm
arrc = np.arange(10)*cm
arrd = np.random.rand(10)*1.123*cm


# %%
print(arra)
print(arrb)
print(arrc)
print(arrd)
print()
print(repr(arra))
print(repr(arrb))
print(repr(arrc))
print(repr(arrd))
print()
print(str(arra))
print(str(arrb))
print(str(arrc))
print(str(arrd))
print()
print((arra)._repr_latex_())
print((arrb)._repr_latex_())
print((arrc)._repr_latex_())
print((arrd)._repr_latex_())



# %%
arra

# %%
arrb

# %%
arrc

# %%
arrd

# %% [markdown]
# ## 2020-10-31 test plot

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp

setup_matplotlib()

mm = units['mm']
km = units["km"]
cm = units["cm"]
print(mm)

arrd = np.random.rand(10)*1.123*cm
arrd.favunit = m

# %%
arrd

# %%
plt.plot(arrd, arrd)

# %%

# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp

#setup_matplotlib()

mm = units['mm']
km = units["km"]
cm = units["cm"]

plt.axhline(2*m)

# %% [markdown]
# ## 2020-11-05 interp

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp

#setup_matplotlib()

mm = units['mm']
km = units["km"]
cm = units["cm"]

# %%
q = np.arange(10)*s
qa = np.arange(20, 30)*m
a = 23.4*m

print(np.interp(a, qa, q))
print(np.interp(0*m, qa, q, 100*s))

try:
    np.interp(0*m, qa, q, 100*s**2)
except Exception as e:
    print(e)


try:
    np.interp(0*s, qa, q)
except Exception as e:
    print(e)

    

# %% [markdown]
# ## 2020-11-06 faire un generateur de qarray a partir d'une liste de q

# %%

# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib
from handcalcs import render
import forallpeople as si

from physipy import Dimension, units, quantify, Quantity
from IPython.display import display, Latex
from sympy.printing import latex
import pint
import sympy as sp

#setup_matplotlib()

mm = units['mm']
km = units["km"]
cm = units["cm"]



# %%

# %%
#q2 = physipy.quantity.utils.array_to_Q_array(a2)
#print(q2)
#q2.value

# %%

# %%
def asqarray(array_like):
    """The value returned will always be a Quantity with array value"""
    if isinstance(array_like, list):
        if isinstance(array_like[0], Quantity):
            dim = array_like[0].dimension
            val_list = []
            for q in array_like:
                if q.dimension == dim:    
                    val_list.append(q.value)
                    res_val = np.array(val_list)
                else:
                    raise DimensionError(q.dim, dim)
            return Quantity(res_val, dim)
        else:
            return quantify(array_like)
    elif isinstance(array_like, np.ndarray):
        if isinstance(array_like[0], Quantity):
            dim = array_like[0].dimension
            val_list = []
            for q in array_like:
                if q.dimension == dim:    
                    val_list.append(q.value)
                    res_val = np.array(val_list)
                else:
                    raise DimensionError(q.dim, dim)
            return Quantity(res_val, dim)
        else:
            return quantify(array_like)
    else:
        raise ValueError("Type {type(array_like)} not supported")



# %%
list_q = [1*s, 2*s]                          # [1 2] s
list_f = [1., 2.]                            # [1. 2.] None
arr_q = np.array([1*s, 2*s] , dtype=object)  # [1 2] s   
arr_mono_q = np.asarray([1*m], dtype=object) # [1] m 
arr_f = np.array([1., 2.]    )               # [1. 2.] None

# %%
print(asqarray(list_q), type(asqarray(list_q)))
print(asqarray(list_f), type(asqarray(list_f)))
print(asqarray(arr_q) , type(asqarray(arr_q) ))
print(asqarray(arr_mono_q) , type(asqarray(arr_mono_q) ))
print(asqarray(arr_f) , type(asqarray(arr_f) ))


# %% [markdown]
# ## 2020-11-11 Widgets

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib
from physipy.qwidgets.qwidgets import QuantityText, FDQuantitySlider

from physipy import Dimension, units, quantify, Quantity

# %%
import ipywidgets as ipyw

from physipy.qwidgets.qwidgets import *

# %% [markdown]
# At init of interactive, https://github.com/jupyter-widgets/ipywidgets/blob/9f6d5de1025fb02e7cad16c0b0cd462614482c36/ipywidgets/widgets/interaction.py#L187
# a loop is done on widgets and check if are DOMWidget. If not, they are not added to the output hence not rendered on the front end ; can be seen in print(w.children) and comparaing with a classic slider
#
# To work, widget must inherit from ValueWidget
#
# So t make it work, I should create real low level widgets inheriting from DOMWIdget https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Low%20Level.html?highlight=DOMWidget#Widget-skeleton
#
# Carefull with abbreviation and iter : iter makes believe Quantity is a tuple or some kind

# %% [markdown]
# ### Widget does not appear when provided to interact

# %%

qw = QuantityText(3*m)

#@ipyw.interact(x=qw)
def print_x(x):
    print(x)
    return x
    
ipyw.interact(print_x, x=QuantityText(3*m))


# %%
def toto(x):
    print(x)
    return x

ipyw.interact(toto, x=ipyw.IntSlider())

# %% [markdown]
# #### Interactive

# %%

qw = QuantityText(3*m)

#@ipyw.interact(x=qw)
def print_x(x):
    print(x)
    return x
    
w = ipyw.interactive(print_x, x=QuantityText(3*m))
w2 = ipyw.interactive(print_x, x=3)


print(isinstance(QuantityText(), ipyw.DOMWidget))

# %%
print(w.kwargs_widgets)
print(w.args)
print(w.kwargs)
print(w.out)
print(w.children)

# %%
print(w2.kwargs_widgets)
print(w2.args)
print(w2.kwargs)
print(w2.out)
print(w2.children)

# %% [markdown]
# #### interactive ouput

# %%

wa = ipyw.IntSlider()
wb = ipyw.IntSlider()
wc = ipyw.IntSlider()

# An HBox lays out its children horizontally
ui = ipyw.HBox([wa, wb, wc])

def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook 
    # output area.
    print((a, b, c))

out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)

# %%
wa = FDQuantitySlider(3*m)
wb = FDQuantitySlider(3*m)
wc = FDQuantitySlider(3*m)

# An HBox lays out its children horizontally
ui = ipyw.HBox([wa, wb, wc])


# %%
def f(a, b, c):
    # You can use print here instead of display because interactive_output generates a normal notebook 
    # output area.
    print((a, b, c))

out = ipyw.interactive_output(f, {'a': wa, 'b': wb, 'c': wc})

display(ui, out)

# %% [markdown]
# #### link : not working

# %%
wa = QuantityText(3*m)
wb = QuantityText(3*m)

wa

# %%
wb

# %%
mylink = ipyw.link((wa, 'value'), (wb, 'value'))

# %%
wa

# %% [markdown]
# #### jslink  : not working

# %%
mylink = ipyw.jslink((wa, 'value'), (wb, 'value'))

# %% [markdown]
# #### observing

# %%
slider = ipyw.FloatSlider(
    value=7.5,
    min=5.0,
    max=10.0,
    step=0.1,
    description='Input:',
)

# Create non-editable text area to display square of value
square_display = ipyw.HTML(description="Square: ",
                              value='{}'.format(slider.value**2))

# Create function to update square_display's value when slider changes
def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)
    
slider.observe(update_square_display, names='value')

# Put them in a vertical box
ipyw.VBox([slider, square_display])


# %%
slider = FDQuantitySlider(
    value=7.5*m)

# Create non-editable text area to display square of value
square_display = ipyw.HTML(description="Square: ",
                              value='{}'.format(slider.value**2))

# Create function to update square_display's value when slider changes
def update_square_display(change):
    square_display.value = '{}'.format(change.new**2)
    
slider.observe(update_square_display, names='value')

# Put them in a vertical box
slider

square_display

# %% [markdown]
# ## 2020-11-13 decorators
#

# %% [raw]
# def arg_is_greater(min_qs):
#     def decorator(func):
#         def decorated(*args, **kwargs):
#             for arg, minq in zip(args, min_qs):
#                 if not arg > minq:
#                     raise ValueError("Arg must be > to ", minq)            
#             res = func(*args, **kwargs)
#             return res
#         return decorated
#     return decorator
#
# def arg_is_less(max_qs):
#     def decorator(func):
#         def decorated(*args, **kwargs):
#             for arg, maxq in zip(args, max_qs):
#                 if not arg > minq:
#                     raise ValueError("Arg must be < to ", maxq)            
#             res = func(*args, **kwargs)
#             return res
#         return decorated
#     return decorator
#
#
# def check_inputs_on_operator(list_of_tuples):
#     """
#     from operator import lt, gt, ne
#
#     @physipy.quantity.utils.check_inputs_on_operator(((lt, 0*m),))
#     def toto(x, y):
#         return x + y
#         
#     Ideas : 
#      - default comparisons to 0 when right is not provided
#      - check if operators are unary or binary and change default behavior
#      - same for outputs
#     """
#     
#     def decorator(func):
#         def decorated(*args, **kwargs):
#             for tup, arg in zip(list_of_tuples, args):
#                 oper, right = tup[0], tup[1]
#                 if not oper(arg, right):
#                     raise ValueError("Arg ", str(arg), " did not met condition", str(oper), "with", str(right))
#             res = func(*args, **kwargs)
#             return res
#         return decorated
#     return decorator   

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib

from physipy import Dimension, units, quantify, Quantity

# %%
from operator import lt, gt, ne

@physipy.quantity.utils.check_inputs_on_operator(((lt, 0*m),))
def toto(x, y):
    return x + y



toto(-3*m, 1*m)

# %% [markdown]
# ## repr on negtiv values

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib

from physipy import Dimension, units, quantify, Quantity

mm = units["mm"]

# %%
-1*m

# %%
-1*mm


# %%
def toto(x, y):
    return x + y

toto(-3*m, 1*m)

# %% [markdown]
# ## 2020-11-13 repr

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib

from physipy import Dimension, units, quantify, Quantity

mm = units["mm"]

# %%
a2 = np.array([1*s, 2*s], dtype=object)
q2 = physipy.quantity.utils.array_to_Q_array(a2)

q2

# %%
print(q2)

# %% [markdown]
# ## 2020-11-14 list of to q

# %%
# restart kernel
import sys
#print(sys.path)
sys.path.insert(0, r"/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/physipy")
import matplotlib.pyplot as plt
import numpy as np

import physipy
from physipy import s, m, setup_matplotlib

from physipy import Dimension, units, quantify, Quantity
from physipy.quantity.utils import asqarray, qarange

mm = units["mm"]

# %%
res = []
for q in qarange(1, 10)*m: 
    print(q)
    res.append(q)
print(res)

# quantify(res) 
asqarray(res)
print(np.array(res, dtype=object))
print(np.asarray(res, dtype=object))
print(np.asanyarray(res, dtype=object))


# %%
import pint

# %%
ureg = pint.UnitRegistry()

# %%
res = [i*ureg.m for i in range(10)]
res

# %%
#np.array(res)
print(np.array(res, dtype=object))
print(np.asarray(res, dtype=object))
print(np.asanyarray(res, dtype=object))

#print(np.array(res))
#print(np.asarray(res))
#print(np.asanyarray(res))

# %% [markdown]
# # .To(1/K)  fails

# %%
from physipy import K, s, units
W = units["W"]

# %%
a = 1*s
b = 2*W
print(a.to(K))
print(b.to(K))

# %%
c = a.to(K)
c._compute_complement_value()


# %%
def into(q, unit, dec):
    """
    into returns a copy with a new favunit
    """
    qq = q.into(unit)
    qq.DIGITS = dec
    return qq


# %%
q = 1.234567*W
q.DIGITS

# %%
qq = into(q, W, 7)
qq.DIGITS
qq._format_value()
#qq._compute_complement_value()
#qq

# %%
import sympy as sp

sp.printing.latex(sp.parsing.sympy_parser.parse_expr(qq._format_value()))

# %%
