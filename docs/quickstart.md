The pitch :

```python
import numpy as np
import matplotlib.pyplot as plt

import physipy
from physipy import units, constants, set_favunit, setup_matplotlib
from physipy import m, kg, K, sr

# reading units and constants
W = units["W"]
mum = units["mum"]
hp = constants["Planck"]
c = constants["c"]
kB = constants["k"]

# create a function, and attach a favorite unit (for display)
@set_favunit(W/(m**2*sr*mum))
def planck_W(wl, T):
    return 2*hp*c**2/(wl**5) * 1/(np.exp(hp*c/(wl*kB*T))-1)/sr

# create scalar with unit
T_bb = 5800*K

# create an array with unit
ech_wl = np.linspace(0.3, 3, 100)*mum 
ech_wl.favunit = mum

# activate favunit handling for automatic plot axis label
setup_matplotlib()

plt.plot(ech_wl, planck_W(ech_wl, T_bb))

```

# A quickstart on physipy
Homepage of project : [physipy](https://github.com/mocquin/physipy)

```python
import numpy as np

import physipy
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
## Dimension object
<!-- #endregion -->

The Dimension object is basically a dictionnary that stores the dimensions' name and power. A dimension can be created different ways :

```python
a_length_dimension = physipy.Dimension("L")
print(a_length_dimension)
a_length_dimension
```

```python
a_speed_dimension = physipy.Dimension({"L": 1, "T":-1})
print(a_speed_dimension)
a_speed_dimension
```

Dimensions can be multiplied and divided as expected : 

```python
product_dim = a_length_dimension * a_speed_dimension
print(product_dim)
product_dim
```

```python
div_dim = a_length_dimension / a_speed_dimension
print(div_dim)
div_dim
```

You can display a dimension in terms of corresponding SI unit (returns a string) :

```python
print(a_length_dimension.str_SI_unit()) # meters
print(a_speed_dimension.str_SI_unit()) # meters/second
```

Other operations are avalaible : 

```python
print((a_length_dimension**2).str_SI_unit())
print(a_length_dimension == a_speed_dimension)
print((1/a_length_dimension).str_SI_unit())
```

## Quantity object


The Quantity class is simply the association of a numerical value, and a dimension. It can be created several ways :

```python
yo_mama_weight = physipy.Quantity(2000, physipy.Dimension("M"))
print(yo_mama_weight)
```

```python
yo_papa_weight = 2000 * physipy.kg
print(yo_papa_weight)
```

```python
print(yo_mama_weight == yo_papa_weight)
```

If dimension analysis allows it, you can perform standard operations on and between Quantity objects :

```python
print(yo_mama_weight + yo_papa_weight)
```

```python
# speed of light
c = physipy.constants["c"]
E_mama = yo_mama_weight * c**2
print(E_mama)
```

## Unit conversion and displaying


You can change the unit a Quantity displays by changing its ```favunit``` attribute, which means "favorite unit". It default to ```None```which displays the Quantity in SI-units.

```python
print(yo_mama_weight.favunit)
```

```python
# displaying in SI-unit, kg
print(yo_mama_weight)
```

```python
# changing the favunit
g = physipy.units["g"]
yo_mama_weight.favunit = g
```

```python
# now displayed in grams
print(yo_mama_weight)
```

Another example : 

```python
speed_of_light = c
print(c)
```

```python
mile = physipy.imperial_units["mil"]
one_hour = physipy.units["h"]
retarded_speed_unit = mile / one_hour
print(c.to(retarded_speed_unit))
```

## Units and constants


Lots of units and constants are packed up in various dicts. The keys are the symbol of the units/constant, and the value is the corresponding quantity.

```python
# pico-Ampere
pA = physipy.units["pA"]
print(pA)
```

```python
# Planck's constant
h_p = physipy.constants["h"] 
print(h_p)
```

Note that units and constants are just Quantity objects !

```python
print(type(pA))
print(type(h_p))
```

## Numpy compatibility


You can define a Quantity with a numpy.ndarray value :

```python
position_sampling = np.array([1,2,3]) * physipy.m
print(position_sampling)
```

```python
time_sampling = physipy.Quantity([0.1, 0.2, 0.3], physipy.Dimension("T"))
print(time_sampling)
```

You can then play with those as you would with regular ndarrays, as long as you respect dimensional analysis :

```python
print(position_sampling / time_sampling)
```

```python
print(2 * position_sampling)
```

```python
try:
    position_sampling + time_sampling
except Exception as e:
    print("You can't add a length and a time dummy !")
    print(e)
```

```python
from math import pi
try:
    # you cant compute the cos of a length
    np.cos(position_sampling)
except:
    # but you can for a plane angle
    an_angle_array = np.array([0, pi/2, pi]) * physipy.rad
    print(np.cos(an_angle_array))
    # it also works with degrees of course
    another_angle_array = np.array([0, 90, 180]) * physipy.units["deg"]
    print(np.cos(another_angle_array))
```

## List of constants and units


### Units

```python
print(physipy.SI_units.keys())
```

```python
print(physipy.SI_derived_units.keys())
```

```python
print(physipy.imperial_units.keys())
```

### Constants

```python
print(physipy.scipy_constants.keys())
```

```python
print(physipy.scipy_constants_codata.keys())
```

```python

```
