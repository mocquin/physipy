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

# %% [markdown]
# # Function UI with widgets and decorators

# %% [markdown]
# It is possible to get a function widget just using decorators : 
#  - if available, a "name" will be displayed
#  - if available, an equation in Latex will be displayed
#  - if available, the annotation will be used as start value for inputs
#  - if available, the annotation for output will used as favunit

# %%
from physipy.quantity.utils import latex_eq, name_eq
from physipy import units, sr, constants, m, s
import numpy as np
from physipy.qwidgets.ui import ui_widget_decorate_from_annotations, ui_widget_decorate

pi = np.pi
g = constants["g"]
mm = units['mm']
msr = units["msr"]


# %% [markdown]
# ## using functions

# %% tags=[]
# define the function, and optionnaly add decorators, and annotations
@latex_eq(r"v = d/t")
@name_eq("Speed")
def speed(length: m, time: s) -> m/s:
    return length / time


# create the ui
speed_ui = ui_widget_decorate_from_annotations(speed)

# display the ui
speed_ui


# %%
# define the function, and optionnaly add decorators, and annotations
@latex_eq(r"v = d/t")
@name_eq("Speed")
def speed(length: m, time: s) -> m/s:
    return length / time

# create the ui
speed_ui = ui_widget_decorate_from_annotations(speed, kind="TextSlider")

# display the ui
speed_ui


# %%
# define the function, and optionnaly add decorators, and annotations
@latex_eq(r"$v = d/t$")
@name_eq("Speed")
def speed(length: m, time: s) -> m/s:
    return length / time


# create the ui
speed_ui = ui_widget_decorate_from_annotations(speed)

# display the ui
speed_ui


# %% [markdown] tags=[]
# ## using decorator notation

# %% [markdown]
# Equivalently using decorator notation

# %%
# equivalently
@ui_widget_decorate_from_annotations
@latex_eq(r"$v = \frac{d}{t}$")
@name_eq("Speed")
def speed(length: 2*m, time: 10*s) -> m/s:
    return length / time


speed


# %% [markdown]
# More examples

# %%
# define the function, and optionnaly add decorators, and annotations
@latex_eq(r"$\Omega = \frac{\pi}{4(f/D)^2}$")
@name_eq("PSA")
def psa(f: m, D: m) -> msr:          # the output will be displayed using msr
    return np.pi / (4*(f/D)**2)*sr


# create the ui
psa_ui = ui_widget_decorate_from_annotations(psa)

# display the ui
psa_ui


# %%
# define the function, and optionnaly add decorators, and annotations
@latex_eq(r"$T = 2\pi \sqrt{\frac{L}{g}}$")
@name_eq("Pendulum period :")
def pendulum_period(L: mm):
    return 2 * pi * np.sqrt(L/g)


# create the ui
pendulum_ui = ui_widget_decorate_from_annotations(pendulum_period)

# display the ui
pendulum_ui

# %% [markdown]
# Without annotations

# %%
ohm = units["ohm"]
farad = units["F"]


@latex_eq(r"$f_c = \frac{1}{2\pi RC}$")
@name_eq("Cut-off frequency of R-C circuit :")
def freq_RC(R, C):
    return 1/(2 * pi * R * C)


freq_RC_ui = ui_widget_decorate([("R", ohm),
                                 ("C", farad, "Capacitance")])(freq_RC)

freq_RC_ui

# %% [markdown]
# With favunit output

# %%
Hz = units["Hz"]
GHz = Hz*10**9
GHz.symbol = "GHz"

from physipy import set_favunit

@ui_widget_decorate_from_annotations
@set_favunit(GHz)
@latex_eq(r"$f_c = \frac{1}{2\pi RC}$")
@name_eq("Cut-off frequency of R-C circuit :")
def freq_RC(R:ohm, C:farad):
    return 1/(2 * pi * R * C)

freq_RC


# %% [markdown]
# Use annotation to set favunit

# %%
@ui_widget_decorate_from_annotations
@latex_eq(r"$f_c = \frac{1}{2\pi RC}$")
@name_eq("Cut-off frequency of R-C circuit :")
def freq_RC(R:ohm, C:farad)->GHz:
    return 1/(2 * pi * R * C)

freq_RC

# %%
