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
# # Fourier transform handling units

# %%
import numpy as np
import physipy
from physipy import units, m, s, setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
import scipy.signal

# %%
V = units["V"]
ms = units["ms"]

# %%
dt = 5*ms
fs = 1/dt
# signal is a noise with unit W
n = 100
sig = 5*np.random.randn(n)*V
sig.favunit = V
sig -= np.mean(sig)
ech_t = np.linspace(0, n-1, num=n)*dt

fig, ax = plt.subplots()
ax.plot(ech_t, sig, "-o")

# %%
print(np.std(sig))
print(np.var(sig))

# %% [markdown]
# Fourier transform of a signal has the same unit : 
# $$X[k] = \sum_n^{N-1} x[k] e^{-i2\pi k n /N }$$

# %%
tf = np.fft.fft(sig, norm="ortho")
print(sig.dimension == tf.dimension)

# %% [markdown]
# Note that several versions of numpy's fft are available : 
#  - `norm=backward` : $X[k]=\sum x[k]e^{-2ikn/N}$
#  - `norm=forward` : $X[k]=\frac{1}{N}\sum x[k]e^{-2ikn/N}$
#  - `norm=ortho`: $X[k]=\frac{1}{\sqrt{N}}\sum x[k]e^{-2ikn/N}$

# %%
sig_test = np.ones(10)
print(np.sum(np.fft.fft(sig_test, norm="backward")))
print(np.sum(np.fft.fft(sig_test, norm="forward")))
print(np.sum(np.fft.fft(sig_test, norm="ortho")))

print(np.fft.fft(sig, norm="ortho").dimension)
print(np.fft.fft(sig, norm="backward").dimension)
print(np.fft.fft(sig, norm="forward").dimension)

# %% [markdown]
# # Modulus
# The modulus of a Fourier transform shall then have the squared unit

# %%
mod2 = tf.real**2 + tf.imag**2#np.real(tf * np.conjugate(tf))
print(mod2.dimension == tf.dimension**2)

# %% [markdown]
# # Frequencies

# %% [markdown]
# Given a sampling period, in second for eg, the sampling frequency is used to compute the frequencies for the fourier transform : 

# %%
dt = 5*s
fs = 1/dt
freqs = np.fft.fftfreq(len(mod2), d=dt)
print(freqs.dimension)
print(freqs)

# %% [markdown]
# # Plotting
# Finally, we can easily plot the spectrum 

# %%
fig, ax = plt.subplots()
mod2.favunit = V**2
ax.plot(freqs, mod2)
print(np.mean(mod2))

# %% [markdown]
# # Frequency shift 

# %%
fig, ax = plt.subplots()
ax.plot(np.fft.fftshift(freqs),
        np.fft.fftshift(mod2))
print(np.mean(mod2))
