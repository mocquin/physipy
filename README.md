# physipy
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mocquin/physipy/HEAD)
[![PyPI version](https://badge.fury.io/py/physipy.svg)](https://badge.fury.io/py/physipy)

This python package allows you to manipulate physical quantities, basically considering in the association of a value (scalar, numpy.ndarray and more) and a physical unit (like meter or joule).

```python
>>> from physipy.quickstart import nm, hp, c, J
>>> E_ph = hp * c / (500 * nm)
>>> print(E_ph)
3.9728916483435158e-19 kg*m**2/s**2
>>> E_ph.favunit = J
>>> print(E_ph)
3.9728916483435158e-19 J
```

For a quickstart, check the [quickstart notebook](https://github.com/mocquin/physipy/blob/master/quickstart.ipynb) on the [homepage](https://github.com/mocquin/physipy)
Get a live session at [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mocquin/physipy/HEAD)

## Installation

```
pip install physipy
```

## Goals

- Few LOC
- Simple architecture, with only 2 classes (namely Dimension and Quantity)
- High numpy compatibility
- Human-readable syntax (fast syntax !)
 
## Use case

- Define scalar and arrays of physical quantities
- Compute operation between them : add, sub, mul, div, pow, and so on
- Display physical quantities in various “units”
 
## Implementation approach

The implementation is pretty simple : 
- a Dimension object represents a [physical dimension](https://en.wikipedia.org/wiki/Dimensional_analysis). For now, these dimension are based on the [SI unit](https://en.wikipedia.org/wiki/International_System_of_Units). It is basically a dictionary where the keys represent the base dimensions, and the values are the exponent these dimensions.
- a Quantity object is simply the association of a value, scalar or array (or more!), and a Dimension object. Note that this Quantity classe does not sub-class numpy.ndarray (although Quantity objects are compatible with numpy's ufuncs). Most of the work is done by this class.
- By default, a Quantity is displayed in term of SI untis. To express a Quantity in another unit, just set the "favunit", which stands for "favourite unit" of the Quantity : ```my_toe_length.favunit = mm```.
- Plenty of common units (ex : Watt) and constants (ex : speed of light) are packed in. Your physical quantities (```my_toe_length```), units (```kg```), and constants (```kB```) are all Quantity objects.


## Numpy's support
See [NEP35](https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html#nep35).

## Known issues

### numpy.full
```python
import numpy as np
from physipy import m
np.full((3,3), 2*m)
```

## About angles and units

See : https://www.bipm.org/en/CGPM/db/20/8/.
Astropy's base units : https://docs.astropy.org/en/stable/units/standard_units.html#enabling-other-units

## Alternative packages

There are plenty of python packages that handle physical quantities computation. Some of them are full packages while some are just plain python module. Here is a list of those I could find (approximately sorted by guessed-popularity) :

 - [astropy](http://www.astropy.org/astropy-tutorials/Quantities.html)
 - [sympy](https://docs.sympy.org/latest/modules/physics/units/philosophy.html)
 - [pint](https://pint.readthedocs.io/en/latest/)
 - [forallpeople](https://github.com/connorferster/forallpeople)
 - [unyt](https://github.com/yt-project/unyt)
 - [Unum](https://bitbucket.org/kiv/unum/)
 - [magnitude](http://juanreyero.com/open/magnitude/)
 -  physics.py : there are actually several packages based on the same core code : [ipython-physics](https://bitbucket.org/birkenfeld/ipython-physics) (python 2 only) and [python3-physics](https://github.com/TheGrum/python3-physics) (python 3 only)
 - [ScientificPython.Scientific.Physics.PhysicalQuantities](https://github.com/ScientificPython/ScientificPython)
 - [numericalunits](https://github.com/sbyrnes321/numericalunits)
 - [dimensions.py](https://code.activestate.com/recipes/577333-numerical-type-with-units-dimensionspy/) (python 2 only)
 - [buckingham](https://github.com/mdipierro/buckingham)
 - [units](https://bitbucket.org/adonohue/units/)
 - [quantities](https://pythonhosted.org/quantities/user/tutorial.html)
 - [physical-quantities](https://github.com/hplgit/physical-quantities)
 - [brian](https://brian2.readthedocs.io/en/stable/user/units.html)
 - [quantiphy](https://github.com/KenKundert/quantiphy)
 - [parampy](https://github.com/matthewwardrop/python-parampy/blob/master/parampy/quantities.pyx)
 - [pynbody](https://github.com/pynbody/pynbody)
 - [python-units](https://pypi.org/project/python-units/)
 - [misu](https://github.com/cjrh/misu)
 - and finally [pysics](https://bitbucket.org/Phicem/pysics) from which this package was inspired

If you know another package that is not in this list yet, feel free to contribute ! Also, if you are interested in the subject of physical quantities packages in python, check this [quantities-comparison](https://github.com/tbekolay/quantities-comparison) repo and [this talk](https://www.youtube.com/watch?v=N-edLdxiM40). Also check this [comparison table](https://socialcompare.com/en/comparison/python-units-quantities-packages).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgment

Thumbs up to phicem and his [pysics](https://bitbucket.org/Phicem/pysics) package, on which this package was highly inspired. Check it out !
