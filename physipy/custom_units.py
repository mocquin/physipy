import sys
sys.path.insert(0,'/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/quantity-master')

from quantity import m, s, kg, A, cd, mol, K, rad, units, make_quantity, SI_units_derived
from math import pi

cm = SI_units_derived["cm"]
g = units['g']
h = units["h"]
J = units["J"]
W = units["W"]
kJ = J * 1000
kJ.symbol = 'kJ'


raw_units = {"deg": pi/180 *rad,
             "liter": 0.001 * m**3
            }

# Extending quantity units dict
for key, value in raw_units.items():
    units[key] = make_quantity(value, symbol=key)



# imperial units from astropy. This is ridiculous...
liter = units["liter"]
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

imperial_units = {}
for key, value in raw_imperial_units.items():
    imperial_units[key] = make_quantity(value, symbol=key)
    
#units = {key, make_quantity(value, symbol=key) for key, value in raw_units.items()}

