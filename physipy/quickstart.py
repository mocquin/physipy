import sys
sys.path.insert(0,'/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/quantity-master/')


import numpy as np
from numpy import pi, cos, arccos, sin, arcsin, tan, arctan, exp, log
from physipy import Quantity, Dimension
from physipy import m, kg, s, A, K, cd, mol, rad, sr, SI_units_derived
from physipy.constants import constants
from physipy.units import units


Hz = units["Hz"]
J = units["J"]
W = units["W"]
h = units["h"]
lm = units["lm"]
lx = units["lx"]

km = SI_units_derived["km"]

hp = constants["h"]
c = constants["c"]
kB = constants["k"]