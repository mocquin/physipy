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
"""

from numpy import pi
from .quantity import Quantity, Dimension, SI_UNIT_SYMBOL, quantify, make_quantity


# Dictionnary of prefixes
PREFIX_DICT = {
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


def CREATE_BASE_SI_UNIT_DICT(prefix_dic, base_units_symbol_dim, dic={}):
    """Create the prefixed dict for the base SI units
    
    Extends the dic by adding the combination between prefix_dic and base_units
    """
    for prefix_symbol, prefix_value in prefix_dic.items():
        for dim_symbol, unit_symbol in base_units_symbol_dim.items():
            prefixed_unit_symbol = prefix_symbol + unit_symbol
            if prefixed_unit_symbol == "mkg":
                dic["g"] = Quantity(0.001, Dimension("M"), symbol="g")
            # Update dic
            dic[prefixed_unit_symbol] = Quantity(prefix_value,Dimension(dim_symbol),symbol=prefixed_unit_symbol)
    return dic


def prefix_units(prefix_dic, unit_dict, extend=False):
    """Return a dict of unit with all combination between the input unit dict and the prefix dict."""
    prefixed_dict = {}
    for prefix_symbol, prefix_value in prefix_dic.items():
        for unit_symbol, unit_quantity in unit_dict.items():
            prefixed_symbol = prefix_symbol+ str(unit_quantity.symbol)
            prefixed_dict[prefixed_symbol] = make_quantity(prefix_value * unit_quantity,
                                                           symbol=prefixed_symbol)
    return prefixed_dict if extend == False else {**unit_dict, **prefix_dic}
    

# Init of SI inits dict
SI_units = {value: Quantity(1,Dimension(key), symbol=value) for (key,value) in SI_UNIT_SYMBOL.items()}


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
SI_units_prefixed = CREATE_BASE_SI_UNIT_DICT(PREFIX_DICT, SI_UNIT_SYMBOL, SI_units) # extends SI_units


# SI derived units
SI_derived_units_raw = {
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

SI_derived_units = {key: make_quantity(value, symbol=key) for key, value in SI_derived_units_raw.items()}
SI_derived_units_prefixed = prefix_units(PREFIX_DICT, SI_derived_units, extend=True)


# Other units
other_accepted_units_raw = {
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

other_units = {key: make_quantity(value, symbol=key) for key, value in other_accepted_units_raw.items()}


# Concatenating units
units = {**SI_units_prefixed, **SI_derived_units, **other_units} #including base SI units to units dict

all_units = {**SI_units_prefixed, **SI_derived_units_prefixed, **other_units}

del pi
del Quantity, Dimension, SI_UNIT_SYMBOL, quantify, make_quantity
del SI_derived_units_raw, other_accepted_units_raw