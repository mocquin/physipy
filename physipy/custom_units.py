# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define non-SI-units.

This module provides for now 2 dictionnaries of units : 
 - custom_units : for user-defined units
 - imperial_units : retard units
 
TODO :
 - create a function wrapper for dict creation ?
 - Should custom units and constants be in the same module ?

"""

# setupe
from math import pi

from .quantity import make_quantity
from .quantity import m, kg, s, A, K, cd, mol, rad, sr, SI_units, SI_units_prefixed, SI_derived_units, other_units, units

cm = SI_units_prefixed["cm"]
g = units['g']
h = units["h"]
J = units["J"]
W = units["W"]
kJ = J * 1000
kJ.symbol = 'kJ'
liter = units["L"]


# Define here you custom units : key=symbol and value=quantity
raw_custom_units = {

}


# imperial units from astropy. This is ridiculous...
raw_imperial_units = {
    # LENGTHS
    "in": 2.54 * cm,
    "ft": 12 * 2.54 * cm,
    "yd": 3 * 12 * 2.54 *cm,
    "mi": 5280  * 12* 2.54*cm,
    "mil": 0.001 * 2.54 * cm,
    "NM": 1852 * m,
    "fur": 660 * 12 * 2.54 * cm,
    # AREAS
    "ac": 43560 * (12 * 2.54*cm)**2,
    # VOLUMES
    'gallon':    liter / 0.264172052,
    'quart':    (liter / 0.264172052) / 4,
    'pint':    ((liter / 0.264172052) / 4) / 2,
    'cup':    (((liter / 0.264172052) / 4) / 2) / 2, 
    'foz':   ((((liter / 0.264172052) / 4) / 2) / 2) / 8, 
    'tbsp': (((((liter / 0.264172052) / 4) / 2) / 2) / 8) / 2, 
    'tsp': ((((((liter / 0.264172052) / 4) / 2) / 2) / 8) / 2) / 3,
    # MASS
    'oz': 28.349523125 * g,
    'lb': 16 * 28.349523125 * g, 
    'st': 14 * 16 * 28.349523125 * g, 
    'ton': 2000 * 16 * 28.349523125 * g, 
    'slug': 32.174049 * 16 * 28.349523125 * g,
    # SPEED
    'kn': 1852 * m / h,
    # FORCE
    'lbf': (32.174049 * 16 * 28.349523125 * g) * (12 * 2.54 * cm) * s**(-2),
    'kip': 1000 * (32.174049 * 16 * 28.349523125 * g) * (12 * 2.54 * cm) * s**(-2), 
    # ENERGY
    'BTU': 1.05505585 * kJ,
    'cal': 4.184 * J,
    'kcal': 1000 * 4.184 * J,
    # PRESSURE
    'psi': (32.174049 * 16 * 28.349523125 * g) * (12 * 2.54 * cm) * s**(-2) * (2.54 * cm) ** (-2),
    # POWER
    'hp': W / 0.00134102209,
    # TEMPERATURE
    # not dealing with farheneiht unit
}


# custom units dict
custom_units = {}
for key, value in raw_custom_units.items():
    custom_units[key] = make_quantity(value, symbol=key)


# imperial unit dict
imperial_units = {}
for key, value in raw_imperial_units.items():
    imperial_units[key] = make_quantity(value, symbol=key)

# cleanup
del pi
del m, kg, s, A, K, cd, mol, rad, sr, SI_units, SI_units_prefixed, SI_derived_units, other_units, units
del cm, g, h, J, W, kJ 