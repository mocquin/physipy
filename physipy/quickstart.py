# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define basic units and constants.

Aims to be used with a star import.
"""

import numpy as np
from numpy import pi, cos, arccos, sin, arcsin, tan, arctan, exp, log, sqrt

from physipy import Quantity, Dimension
from physipy import m, kg, s, A, K, cd, mol, rad, sr, units, SI_units_prefixed, SI_derived_units, other_units
from physipy.constants import constants

Hz = units["Hz"]
J = units["J"]
W = units["W"]
h = units["h"]
lm = units["lm"]
lx = units["lx"]

km = SI_units_prefixed["km"]
nm = SI_units_prefixed["nm"]

hp = constants["h"]
c = constants["c"]
kB = constants["k"]