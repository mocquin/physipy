"""
TODO : 
 - [ ] : should 'SI_units_derived' have another name ?
 - [X] : deal with prefixed kg
 - [ ] : add other dicts of units : imperial, astrophys
 - [ ] : add scipy.constants.physical_constants units
 
Questions : 
 - should the definition of other packages units be fixed or relative ?
 - should other packages units/constants be in the same dict ?
"""

from numpy import pi
from .quantity import Quantity, Dimension, SI_UNIT_SYMBOL, quantify, make_quantity


DICT_OF_PREFIX_UNITS = {'Y': 1e24,
                        'Z': 1e21,
                        'E': 1e18,
                        'P': 1e15,
                        'T': 1e12,
                        'G': 1e9,
                        'M': 1e6,
                        'k': 1e3,
                        'h': 1e2,
                        'da': 1e1,
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

def derive_units(prefix_dic, base_units, dic={}):
    """Return all the combination Quantities between the prefixes and the units."""
    for prefix_symbol, prefix_value in prefix_dic.items():
        for dim_symbol, unit_symbol in base_units.items():
            # dealing with kg
            if unit_symbol == "kg" and dim_symbol == "M" :
                unit_symbol = "g"
                prefix_value = prefix_value/1000
            prefixed_unit_symbol = prefix_symbol + unit_symbol
            dic[prefixed_unit_symbol] = Quantity(prefix_value, Dimension(dim_symbol), symbol=prefixed_unit_symbol)
    return dic

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
# dealing with kg for the prefixes
SI_units_derived = derive_units(DICT_OF_PREFIX_UNITS, SI_UNIT_SYMBOL, SI_units)
SI_units_derived["g"] = Quantity(0.001, Dimension("M"), symbol="g")

# Units
units_raw = {"Hz"  : 1/s,
             "N"   : m * kg * s**-2,
             "Pa"  : kg * m**-1 * s**-2,
             "J"   : m**2 * kg * s**-2,
             "W"   : m**2 * kg * s**-3,
             "C"   : s * A,
             "V"   : m**2 * kg * s**-3 * A**-1,
             "F"   : m**-2 * kg**-1 * s**4 * A**2,
             # Omega ? m**2 * kg * s**-3 *A**-2
             "Wb"  : m**2 * kg * s**-2 * A**-1,
             "T"   : kg *s**-2 * A**-1,
             "H"   : m**2 * kg * s**-2 * A**-2,
             "lm"  : cd * sr,
             "lx"  : cd * m**-2,
             "Bq"  : 1/s,
             "h"   : 3600*s,
             "deg" : pi/180 *rad,
             "liter": 0.001 * m**3
            }

units = {key: make_quantity(value, symbol=key) for key, value in units_raw.items()}

units = {**units, **SI_units} #including base SI units to units dict

