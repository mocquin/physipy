# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Physipy can be coupled to many usefull package to have a great python physics framework

# %% [markdown]
# # Handcalcs

# %% [markdown]
# [Handcalcs repo on Github](https://github.com/connorferster/handcalcs)

# %%
# #!pip install --upgrade handcalcs

# %%
from handcalcs import render
import handcalcs
from physipy import m, s, K, units, kg
mm = units["mm"]
import physipy

# %%
# %%render 
b = 2*s

# %%
# %%render
alpha_init = 5*mm # Greek letter

# %%
# %%render
beta_prime = alpha_init**2 + (45*mm)**2 # analytical->numerical->result

# %%
# %%render
Gamma_i = (beta_prime*beta_prime*beta_prime*beta_prime) # only results with paren

# %%
from numpy import sin, sqrt, pi


# %%
def toto(x):
    return 2*x


# %%
# %%render
b = sin(30)
a = sqrt(b**2+pi/sqrt(2/100))
c = sum((1, 2, 3, 4))
d = toto(4*s)

# %%
# %%render
if Gamma_i > 0*m**8: toto=1*kg
else: toto=0*m**2

# %%
from physipy import quad
def tata(x):
    return x


# %%
# %%render
a = 0*m
b = 10*m
res = quad(tata, a, b)

# %%
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from numpy import exp
from physipy import quad, s, m, sr, K, units, constants

# physical constants
h_p = constants["h"]
c = constants["c"]
k_B = constants["k"]
sigma = constants["Stefan_Boltzmann"]

nm = units["nm"]
mum = units["mum"]

def planck(lmbda):
    return 2*h_p*c**2/lmbda**5 * 1/(exp( (h_p * c) / (k_B * 300*K * lmbda)-1)) / sr

lmbda_start = 0.001*nm
lmbda_stop = 1000*mum

# %%
# %%render
res= quad(planck, lmbda_start, lmbda_stop)

# %%
from handcalcs import handcalc


# %%
@handcalc()
def f(x, y, z):
    a = 2*x + 1 * m
    b = y**2 + 100 * s**2
    c = sqrt(z * y)
    return a, b, c


# %%
latex_code, vals_dict = f(1*m, 2*s, 3)

# %%
print(latex_code)

# %% language="latex"
# \begin{aligned}
# a &= 2 \cdot x + 1 \cdot m  = 2 \cdot 1.0\,m + 1 \cdot 1.0\,m &= 3.0\,m  
# \\[10pt]
# b &= \left( y \right) ^{ 2 } + 100 \cdot \left( s \right) ^{ 2 }  = \left( 2.0\,s \right) ^{ 2 } + 100 \cdot \left( 1.0\,s \right) ^{ 2 } &= 104.0\,s^{2}  
# \\[10pt]
# c &= \sqrt { z \cdot y }  = \sqrt { 3 \cdot 2.0\,s } &= 2.45\,s^{0.5}  
# \end{aligned}

# %%
print(vals_dict)


# %%
@handcalc(jupyter_display=True)
def f(x, y, z):
    a = 2*x + 1 * m
    b = y**2 + 100 * s**2
    c = sqrt(z * y)
    return a, b, c


# %%
vals_dict = f(1*m, 2*s, 3)

# %%
