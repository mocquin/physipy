To run the tests : 
 - cd to physipy-master
 - python -m unittest discover

add a quick start import

 
Other module/package :
 - [X] : magnitude : http://juanreyero.com/open/magnitude/
 -  physics.py : 
  - (2.) birkenfeld physics.py (==George Brandl) : ipython-physics : https://bitbucket.org/birkenfeld/ipython-physics
  - [X] : (3.) : python3-physics : https://github.com/TheGrum/python3-physics
 - ScientificPython.Scientific.Physics.PhysicalQuantities (Konrad Hinsen) :
  - (2.7) https://bitbucket.org/khinsen/scientificpython : http://dirac.cnrs-orleans.fr/ScientificPython = https://github.com/ScientificPython/ScientificPython
 - numericalunits : https://github.com/sbyrnes321/numericalunits
 - [ ] : Unum : https://bitbucket.org/kiv/unum/ (https://pypi.org/project/Unum/4.1.0/)
 - (2.) dimensions.py : https://code.activestate.com/recipes/577333-numerical-type-with-units-dimensionspy/
 - units (Aran Donohue) :  https://bitbucket.org/adonohue/units/src
 - https://pythonhosted.org/quantities/user/tutorial.html
 - astropy : http://www.astropy.org/astropy-tutorials/Quantities.html
    - http://learn.astropy.org/rst-tutorials/quantities.html
 - sympy.physics : https://docs.sympy.org/latest/modules/physics/units/philosophy.html
 - https://github.com/hplgit/physical-quantities
 - https://github.com/KenKundert/quantiphy
 - https://quantiphy.readthedocs.io/en/stable/index.html
 - pint : https://pint.readthedocs.io/en/latest/
 - pynbody : https://github.com/pynbody/pynbody
 -  : http://www.southampton.ac.uk/~fangohr/blog/physical-quantities-numerical-value-with-units-in-python.html

 - [ ] : module physics.py : ipython-physics.py : 
Adaptation python 3 de , lui même adpaté de scientificpython de KonradHinsen
http://www.southampton.ac.uk/~fangohr/blog/physical-quantities-numerical-value-with-units-in-python.html
Les "unités" peuvent avoir un "nom" sous forme de chaine de caractères
méthodes tan, cos, sin pour "numpy ufunc" ?
globale précision de 8 par défaut dans la classe
système si avec rad et sr
système cgs dispo
+ extension Ipython pour écriture "1 m"
- [ ] : scientificpython par Konrad Hinsen 



# Known issues

# Advantages:
- Full object-oriented approach : change value attribute, change display
 
# Drawbacks (that could be implemented later ?)
- No simple way to change base-unit system
- No offset scaling (ex degC to K)
- Dimension powers are treated as scalars – one could need them to be treated as rational fractions
- No compatibility with uncertainties
- No compatibility with Fractions
 
# Goals :
- Few LOC
- High numpy compatibility
- Array-like behaviour
- SI-unit base (including for printing)
- Simple syntax (fast syntax !)
 
# Use case :
- Define scalar and arrays of physical quantities
- Compute operation between them : add, sub, mul, div, pow
- Display physical quantities in various “units”.
- Easy ability to use scalar or arrays (in functions) : easy vectorisation
 
# Implementation approach and key mechanic:
- Dimension object represents only the dimension, based on SI-unit. Stored as a dictionary where key is a string of SI-unit, and value is exponent of dimension.
- Quantity object is association of value (scalar or array) and dimension object (Container approach, not subclass of np.ndarray, see TrevorBekolay)
- By default, representation is in SI-unit. To express in any other unit, use the favunit attribute.
- Physical units (ex : Watt), physical constants (ex : speed of light), and physical quantities (ex : 74 kg) are all Quantity objects.
- If we a looking for a physical quantity package, then dimension analysis is needed !
 
# Comparing:
- LOC
- Dinstinguishes between constants, units, dimension, and quantities ?
- List of available constants / units / prefixes ?
- Possibility to declare list of prefixed unit ?
- List of methods implemented for main quantity object
- Presence of some kind of “unit database” ?
- How are dimensionless units handled ? (deg, rad, sr)
- Bench
o    Creation :
§  Constructor
§  Multply a number with another quantity/unit.
o    Repr
§  5 m / 5 meters / 5 length / 5
o    Implementation of units :
§  m / meter / “m” / “meter” / length / “length”
o    Numbers compatibility :
§  Scalars : declaration, print
§  Arrays : declaration, print
o    Operations :
§  Unary :
·         Abs / neg / pos
§  Binary;
·         Add / div /eq / floordiv / ge /gt / le /lt / mod / mul / ne / pow / sub / truediv
·         Same unit
·         Same dimension
·         Different dimension (hence, different unit)
o    Ufunc :
§  Unary
§  Binary
·         Same unit
·         Same dimension
·         Different dimension (hence, different unit)
o    Other numpy functions
§  Argsort / concatenate / mean / median / sort / std / where
o    Time
§  Comparison to pure python (and numpy – no units)
o    Plotting a function on a given segment
§  Ex : plot kinetic energy of a 20kg mass for speed between 0 and 20 km/h
·         Time
·         LOC
 

# Requirements


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


