# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define unit dictionnaries.

This module defines dictionnaries of units :
 - SI_units : all the seven base SI units
 - SI_units_prefixed : same, with the prefixed version
 - SI_derived_units : other accepted units
 - SI_derived_units_prefixed : same, with the prefixed version
 - other_units : other various units
 - units : all of the above

TODO :
 - [ ] : should 'SI_units_derived' have another name ?
 - [X] : deal with prefixed kg
 - [ ] : add other dicts of units : imperial, astrophys
 - [ ] : add scipy.constants.physical_constants units

Questions :
 - should the definition of other packages units be fixed or relative ?
 - should other packages units/constants be in the same dict ?
 - should make a proper data structure, where a new unit added is checked if already taken ?

 Define non-SI-units.

This module provides for now 2 dictionnaries of units :
 - custom_units : for user-defined units
 - imperial_units : retard units

TODO :
 - create a function wrapper for dict creation ?
 - Should custom units and constants be in the same module ?

"""
from __future__ import annotations

from math import pi
from .quantity import Quantity, Dimension, SI_UNIT_SYMBOL, quantify, make_quantity


# Dictionnary of prefixes
_PREFIX_DICT = {
    'Y': 1e24,
    'Z': 1e21,
    'E': 1e18,
    'P': 1e15,
    'T': 1e12,
    'G': 1e9,
    'M': 1e6,
    'k': 1e3,
    'h': 1e2,
    'da': 1e1,
    # skipping base unit
    'd': 1e-1,
    'c': 1e-2,
    'm': 1e-3,
    'mu': 1e-6,
    'n': 1e-9,
    'p': 1e-12,
    'f': 1e-15,
    'a': 1e-18,
    'z': 1e-21,
    'y': 1e-24,
}


def _CREATE_BASE_SI_UNIT_DICT(
        prefix_dic: dict, base_units_symbol_dim: dict, dic: dict = {}) -> dict:
    """
    Extends the dic by adding the combination between prefix_dic and base_units.

    Parameters
    ----------
    prefix_dic : dict
        Dict with keys the string representation of a prefix, and values the corresponding value.
    base_units_symbol_dim : dict
        Dict with keys the dimension symbol, and values the unit symbol.
    dic : dict, optionnal
        Dict on which are added the prefixed values. Default to {}.

    Returns
    -------
    dict
        Dict with keys the prefixed-unit symbol, and values the corresponding Quantity.
    """
    for prefix_symbol, prefix_value in prefix_dic.items():
        for dim_symbol, unit_symbol in base_units_symbol_dim.items():
            prefixed_unit_symbol = prefix_symbol + unit_symbol
            # handle the gram, which is a milli-kilogram
            if prefixed_unit_symbol == "mkg":
                # loop-invariant but should not be executed so not a
                # speed-loss
                dic["g"] = Quantity(0.001, Dimension("M"), symbol="g")
                continue
            # Update dic
            dic[prefixed_unit_symbol] = Quantity(prefix_value,
                                                 Dimension(dim_symbol),
                                                 symbol=prefixed_unit_symbol)
    return dic


def prefix_units(prefix_dic: dict, unit_dict: dict,
                 extend: bool = False) -> dict:
    """Return a dict of unit with all combination between the input unit dict and the prefix dict.

    Parameters
    ----------
    prefix_dic : dict
        Dict with keys the string representation of a prefix, and values the corresponding value.
    unit_dict : dict
        Dict with keys the string of units, and value the corresponding Quantity.
    extend : bool
        Weither to extend the input unit dict, or return the prefixed units in a separate dict

    Returns
    -------
    dict
        Dict with keys the string of prefixed units, and values the corresponding
        Quantity. Extends the input unit_dict if extend is True.
    """
    prefixed_dict = {}
    for prefix_symbol, prefix_value in prefix_dic.items():
        for unit_symbol, unit_quantity in unit_dict.items():
            prefixed_symbol = prefix_symbol + str(unit_quantity.symbol)
            prefixed_dict[prefixed_symbol] = make_quantity(
                prefix_value * unit_quantity, symbol=prefixed_symbol)
    return prefixed_dict if extend == False else {**unit_dict, **prefix_dic}


def _make_quantity_dict_with_symbols(dic):
    return {key: make_quantity(value, symbol=key)
            for key, value in dic.items()}


# Init of SI inits dict
SI_units = {value: Quantity(1, Dimension(key), symbol=value)
            for (key, value) in SI_UNIT_SYMBOL.items()}


kg  = SI_units["kg"]
m   = SI_units["m"]
s   = SI_units["s"]
cd  = SI_units["cd"]
A   = SI_units["A"]
K   = SI_units["K"]
mol = SI_units["mol"]
rad = SI_units["rad"]
sr  = SI_units["sr"]

# Derived SI units with all prefixes
SI_units_prefixed = _CREATE_BASE_SI_UNIT_DICT(
    _PREFIX_DICT, SI_UNIT_SYMBOL, SI_units)  # extends SI_units


# SI derived units
_SI_derived_units_raw = {
    "Hz"  : 1/s,
    "N"   : m * kg * s**-2,
    "Pa"  : kg * m**-1 * s**-2,
    "J"   : m**2 * kg * s**-2,
    "W"   : m**2 * kg * s**-3,
    "C"   : s * A,
    "V"   : m**2 * kg * s**-3 * A**-1,
    "F"   : m**-2 * kg**-1 * s**4 * A**2,
    "ohm" : m**2 * kg * s**-3 *A**-2,
    "S"   : kg**-1 * m**-2 * s**3 * A**2,
    "Wb"  : m**2 * kg * s**-2 * A**-1,
    "T"   : kg *s**-2 * A**-1,
    "H"   : m**2 * kg * s**-2 * A**-2,
    "lm"  : cd * sr,
    "lx"  : cd * m**-2,
    "Bq"  : 1/s,
    "Gy"  : m**2 * s**-2,
    "Sv"  : m**2 * s**-2,
    "kat" : mol * s**-1,
    }
# create the actual dict of units, with symbols
SI_derived_units = _make_quantity_dict_with_symbols(_SI_derived_units_raw)
SI_derived_units_prefixed = prefix_units(
    _PREFIX_DICT, SI_derived_units)


# Other units
_other_accepted_units_raw = {
    "min" : 60 * s,
    "h"   : 3600 * s,
    "d"   : 86400 * s,
    "au"  : 149597870700 * m,
    # how to deal with degree minutes seconds ?
    "deg" : pi/180 *rad,
    "ha"  : 10**4 * m**2,
    "L"   : 10**-3 * m**3,
    "t"   : 1000 * kg,
    "Da"  : 1.660539040 * 10**-27 *kg,
    "eV"  : 1.602176634 * 10**-19 * kg * m**2 * s**-2,
}

other_units = _make_quantity_dict_with_symbols(_other_accepted_units_raw)


# Concatenating units
# including base SI units to units dict
units = {
    **SI_units, 
    **SI_units_prefixed,
    **SI_derived_units,
    **other_units,
    **SI_derived_units_prefixed,
}


cm = SI_units_prefixed["cm"]
g = units['g']
h = units["h"]
J = units["J"]
W = units["W"]
kJ = J * 1000
kJ.symbol = 'kJ'
liter = units["L"]


# imperial units from astropy. This is ridiculous...
raw_imperial_units = {
    # LENGTHS
    "in": 2.54 * cm,
    "ft": 12 * 2.54 * cm,
    "yd": 3 * 12 * 2.54 * cm,
    "mi": 5280 * 12 * 2.54*cm,
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


# imperial unit dict
imperial_units = {key: make_quantity(value, symbol=key)
                  for key, value in raw_imperial_units.items()}
