This page gives you a quick tour of physipy possibilities. For installation instructions, see [here](installation.md).

## Quickstart with physipy by examples

### Example 1 : body-mass index

The simplest way to use physipy is simply to import some units and write your code
using those imported units - thats why most of the time you'll use `from physipy import ...` (and not `import physipy`)`.

Here is an example : 
```python
from physipy import kg, m

# define physical quantities that are unit aware
my_weight = 75 * kg
my_height = 1.89 * m

def bmi_calculator(weight, height):
    "Compute Body-Mass-Index from weight and height."
    return weight / height**2

print(bmi_calculator(my_weight, my_height))
# --> 20.99605274208449 kg/m**2
```

Thanks to physipy, you can use other units like centimeter and gram to define variables, and the result will still be the same:
```python
from physipy import units

# instead of importing kg and m, we import `units`
# which is a dict that contains unit's name as keys and quantity as values
cm = units['cm'] # in the background, cm is just stored as 1/100 of a meter
g  = units['g']   # in the background, g is just stored as 1/1000 of a kilogram

# define physical quantities that are unit aware
my_weight = 75000 * g
my_height = 189 * cm

def bmi_calculator(weight, height):
    "Compute Body-Mass-Index from weight and height."
    return weight / height**2

print(bmi_calculator(my_weight, my_height))
# --> 20.99605274208449 kg/m**2
```

### Example 2 : Kinetic energy
Say you want to compute the kinetic energy of some planets of our solar system using  

$$E_K = \frac{1}{2}m v^2$$

Using list notation or numpy, we can define the mass and speed of 3 planets : 
```python
import numpy as np
from physipy import units, m, kg, s

# we'll do mercury, venus, mars and the earth
masses = [
    3.301 * 10**31, # mercury's mass
    4.867 * 10**24, # venus'
    5.972 * 10**31, # mars'
    6.441 * 10**23, # earth
] * kg

# pop-up the kilometer
km = units['km']

# same order as for the masses
speeds =[47.9, 35, 29.8, 24.1] * km/s

kinetics = 1/2 * masses * speeds**2
# extract the mega-joule unit 
MJ = units['MJ']
kinetics.favunit = MJ
print(kinetics) 
# [3.78692370e+34 2.98103750e+27 2.65168744e+34 1.87049860e+26] MJ
```

### Example 3 : Plotting Planck's law

In this example, we use :

 - physipy units system to attach physical units to numerical quantities
 - the `set_favunit` decorator to apply a 'display' unit to the output of our function
 - the interface with matplotlib, activated using `setup_matplotlib()` so our plot are unit aware

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

## Quickstart on how physipy works

```python
import numpy as np
import physipy
```

### Dimension object

The `Dimension` class is at the core and base of physipy. A `Dimension` object 
is basically a dictionnary that stores the dimensions' name and power. When using physipy, you'll most likely never need to create such objects - but to undertstand how it works, here are some ways to create a dimension 

```python
a_length_dimension = physipy.Dimension("L") # "L" represents the 'length' dimension, associated with the SI unit meter "m"
print(a_length_dimension)
# L

a_speed_dimension = physipy.Dimension({"L": 1, "T":-1}) # "T" represents the 'time' dimension, associated with the SI unit meter "s"
print(a_speed_dimension)
# L/T
```

Dimensions can be multiplied and divided as expected : 

```python
product_dim = a_length_dimension * a_speed_dimension
print(product_dim)
# L**2/T

div_dim = a_length_dimension / a_speed_dimension
print(div_dim)
# T
```

You can display a dimension in terms of corresponding SI unit (returns a string) :

```python
print(a_length_dimension.str_SI_unit()) # meters
print(a_speed_dimension.str_SI_unit())  # meters/second
```

Other operations are avalaible : 

```python
print((a_length_dimension**2).str_SI_unit())    # m**2
print(a_length_dimension == a_speed_dimension)  # False
print((1/a_length_dimension).str_SI_unit())     # 1/m
```

The list of available dimensions is hard-coded (for now) as a text file, and its content is stored in : 
```python
from physipy.quantity.dimension import SI_UNIT_SYMBOL
SI_UNIT_SYMBOL

{'L': 'm', 
 'M': 'kg',
 'T': 's', 
 'I': 'A', 
 'theta': 'K',
 'N': 'mol',
 'J': 'cd',
 'RAD': 'rad',
 'SR': 'sr'}
```

This 'raw' base unit dict is then converted to a dict of variables that implement the Dimension logic in : 
```python
from physipy import SI_units
SI_units

{'m': <Quantity : 1 m, symbol=m>,
 'kg': <Quantity : 1 kg, symbol=kg>,
 's': <Quantity : 1 s, symbol=s>,
 'A': <Quantity : 1 A, symbol=A>,
 'K': <Quantity : 1 K, symbol=K>,
 'mol': <Quantity : 1 mol, symbol=mol>,
 'cd': <Quantity : 1 cd, symbol=cd>,
 'rad': <Quantity : 1 rad, symbol=rad>,
 'sr': <Quantity : 1 sr, symbol=sr>}
```
as you can see, the keys of this dict are the SI short-name, and the values are `Quantity` objects (we will see the `Quantity` just below) with a value of 1 and the corresponding unit stored as a Dimension object.

For example, you can access the Dimension of the meter using the `.dimension` attribute : 
```python
print(SI_units['m'].dimension)
L
```

The above dimension is just wraps a dict whose keys are the SI notation for all dimensions, and the values are the "exponent" for this dimension : 
```python
SI_units['m'].dimension
<Dimension : {'L': 1, 'M': 0, 'T': 0, 'I': 0, 'theta': 0, 'N': 0, 'J': 0, 'RAD': 0, 'SR': 0}>
```

Hence, we see that the dimension of the meter `m` has 1 for length dimension, and 0s for all other dimensions. The complete list is given by the SI-unit system and contains :  

 - `L` : lenght, for a meter `m`
 - `M` :  mass, for a kilogram `kg`
 - `T` : time, for a second `s`
 - `I` : electrical intensity, for an ampère `A`
 - `theta` : thermodynamic temperature, for a Kelvin `K`
 - `N` : quantity of matter, for a mole `mol`
 - `J` : luminous intensity, for a candela `cd`
 - `RAD` : plane angle, for a radian `rad`
 - `SR` : solid angle, for a steradian `sr`
 
For more information, see [the wikipedia page for the international system of units "SI-units"](https://en.wikipedia.org/wiki/International_System_of_Units) and [the wikipedia page for the SI base units](https://en.wikipedia.org/wiki/SI_base_unit).

### Quantity object

Now that you know what a `Dimension` object is and represents, we can use it to define physical quantities using another kind of object : `Quantity`. Note that `Dimension` and `Quantity` are the two and only classes that implement the physical unit logic in physipy, the rest is just boilerplate and numpy compatibility.

The `Quantity` class is simply the association of a numerical value, and a dimension. It can be created 2 ways :

- using the class creator : you won't use this notation much : 

```python
yo_mama_weight = physipy.Quantity(2000, physipy.Dimension("M"))
print(yo_mama_weight)
```
 - using multiplication to attach units to a numerical quantity : 
```python
yo_papa_weight = 2000 * physipy.kg
print(yo_papa_weight)
```

Let's quickly check that both creations lead to the same amount of mass : 
```python
print(yo_mama_weight == yo_papa_weight)
```
A `Quantity` wraps mainly 2 objects of interest : 

 - a `value` that contains the numerical amount, usually a `float` or `np.ndarray`
 - a `dimension` that contains the `Dimension` object to represent its physical dimension

For example : 
```python
print(yo_papa_weight.value)     # 2000
print(yo_papa_weight.dimension) # M
```
As stated above, remember that a 'unit' is just a regular quantity, so we can also inspect it value and dimension : 
```python
print(kg.value)     # 1
print(kg.dimension) # M
```

If dimension analysis allows it, you can perform standard operations on and between `Quantity` objects :

```python
print(yo_mama_weight + yo_papa_weight)
```

If dimension analysis does not allow it, you get a `DimensionError` - remember your teacher explaning you cannot add carrots with potatoes :

```python
5 * m + 3 * s
# raises DimensionError
```

You can then start to compute more complex physical quantities, here is my energy computed using Einstein famous equation $E = m c^2$ : 

```python
# my mass
mass = (75*kg)
# speed of light
c = physipy.constants["c"]
# my energy
energy_myself = mass * c**2

print(energy_myself)
# 6.740663840526132e+18 kg*m**2/s**2
```

### Favorite unit and symbol

It works, but energy quantities are usually not displayed in units of `kg*m**2/s**2`, most likely in joules `J = 1 kg*m**2/s**2` or in electron-Volt `eV = 1.602176634×10−19 J`. Both of these units are available in physipy (more on units below), so let's make use of this to display the energy with a more user-friendly notation. 

To do this, we can set the `favunit` attribute of our energy `Quantity` : the `favunit` must be `None` or another `Quantity` that has a `symbol`.

If the favunit is `None` - the default - a `Quantity` is displayed using SI units, like the energy in the previous example is displayed in `kg*m**2/s**2`. If a `favunit` is set, it will be used on display.

```python
from physipy import units
eV = units['eV']
J = units['J']

# no favunit -> use SI-units
print(energy_myself)
# 6.740663840526132e+18 kg*m**2/s**2

energy_myself.favunit = J
print(energy_myself)
# 6.740663840526132e+18 J

energy_myself.favunit = eV
print(energy_myself)
# 4.2071914528533386e+37 eV
```

Remember that `favunit` absolutely does not change the `Quantity` value or dimension : remember that in the background, the data is stored using a numerical value along with the associated dimension, in this case : 

```python
print(energy_myself.value)
print(energy_myself.dimension)
```

I could define another variable that stores the same physical quantity, but with different favourite units - those variables are hence equal :

```python
energy_myself = mass * c**2
same_but_different = mass * c**2
same_but_different.favunit = eV

print(energy_myself)                        # 6.740663840526132e+18 kg*m**2/s**2
print(same_but_different)                   # 4.2071914528533386e+37 eV
print(energy_myself == same_but_different)  # True
```

### Units and constants

Units and constants are just associations between a specific value and a specific dimension, so physipy implements and exposes
them in a dict using their name-notation as key, and the associate `Quantity` object as value.

#### Units

Units are just conventional names, value and dimension that every body agrees on. 
The most basic units physipy defines are the SI units : 

```python
list(SI_units.keys())
# ['m', 'kg', 's', 'A', 'K', 'mol', 'cd', 'rad', 'sr']
```

From those SI units, many other units can be derived ; again those are just a convention that everybody agrees for the association of a name (or notation), a value and a dimension :  

 - their "prefixed" version : like mm is a prefixed m (0.001 m) or kA is a prefixed ampere (1000 A)
 - other units derived from the SI units : their values are 1, but their dimension is a combination of the base SI units using integer powers : like the newton is `1 N = 1 kg.m.s^-2` is the combination of powers of 1 kilogram, 1 meter, and -2 second
 - anything else : with respect to the SI unit, imperial units are again just the association of a name, a value (that is not 1), and a dimension (that can have anything as their power) : like an inch is 0.0254 meter or a horse-power is about 745.7 kg*m**2/s**3

For those reasons, physipy wraps these units in specific dict : here for the prefixed SI units : 

```python
print(physipy.SI_derived_units.keys())
['m', 'kg', 's', 'A', 'K', 'mol', 'cd', 'rad', 'sr', 'Ym', 'Ykg', 'Ys', 'YA', 'YK', 'Ymol', 'Ycd', 'Yrad', 'Ysr', 'Zm', 'Zkg', 'Zs', 'ZA', 'ZK', 'Zmol', 'Zcd', 'Zrad', 'Zsr', 'Em', 'Ekg', 'Es', 'EA', 'EK', 'Emol', 'Ecd', 'Erad', 'Esr', 'Pm', 'Pkg', 'Ps', 'PA', 'PK', 'Pmol', 'Pcd', 'Prad', 'Psr', 'Tm', 'Tkg', 'Ts', 'TA', 'TK', 'Tmol', 'Tcd', 'Trad', 'Tsr', 'Gm', 'Gkg', 'Gs', 'GA', 'GK', 'Gmol', 'Gcd', 'Grad', 'Gsr', 'Mm', 'Mkg', 'Ms', 'MA', 'MK', 'Mmol', 'Mcd', 'Mrad', 'Msr', 'km', 'kkg', 'ks', 'kA', 'kK', 'kmol', 'kcd', 'krad', 'ksr', 'hm', 'hkg', 'hs', 'hA', 'hK', 'hmol', 'hcd', 'hrad', 'hsr', 'dam', 'dakg', 'das', 'daA', 'daK', 'damol', 'dacd', 'darad', 'dasr', 'dm', 'dkg', 'ds', 'dA', 'dK', 'dmol', 'dcd', 'drad', 'dsr', 'cm', 'ckg', 'cs', 'cA', 'cK', 'cmol', 'ccd', 'crad', 'csr', 'mm', 'g', 'ms', 'mA', 'mK', 'mmol', 'mcd', 'mrad', 'msr', 'mum', 'mukg', 'mus', 'muA', 'muK', 'mumol', 'mucd', 'murad', 'musr', 'nm', 'nkg', 'ns', 'nA', 'nK', 'nmol', 'ncd', 'nrad', 'nsr', 'pm', 'pkg', 'ps', 'pA', 'pK', 'pmol', 'pcd', 'prad', 'psr', 'fm', 'fkg', 'fs', 'fA', 'fK', 'fmol', 'fcd', 'frad', 'fsr', 'am', 'akg', 'as', 'aA', 'aK', 'amol', 'acd', 'arad', 'asr', 'zm', 'zkg', 'zs', 'zA', 'zK', 'zmol', 'zcd', 'zrad', 'zsr', 'ym', 'ykg', 'ys', 'yA', 'yK', 'ymol', 'ycd', 'yrad', 'ysr', 'Hz', 'N', 'Pa', 'J', 'W', 'C', 'V', 'F', 'ohm', 'S', 'Wb', 'T', 'H', 'lm', 'lx', 'Bq', 'Gy', 'Sv', 'kat', 'min', 'h', 'd', 'au', 'deg', 'ha', 'L', 't', 'Da', 'eV', 'YHz', 'YN', 'YPa', 'YJ', 'YW', 'YC', 'YV', 'YF', 'Yohm', 'YS', 'YWb', 'YT', 'YH', 'Ylm', 'Ylx', 'YBq', 'YGy', 'YSv', 'Ykat', 'ZHz', 'ZN', 'ZPa', 'ZJ', 'ZW', 'ZC', 'ZV', 'ZF', 'Zohm', 'ZS', 'ZWb', 'ZT', 'ZH', 'Zlm', 'Zlx', 'ZBq', 'ZGy', 'ZSv', 'Zkat', 'EHz', 'EN', 'EPa', 'EJ', 'EW', 'EC', 'EV', 'EF', 'Eohm', 'ES', 'EWb', 'ET', 'EH', 'Elm', 'Elx', 'EBq', 'EGy', 'ESv', 'Ekat', 'PHz', 'PN', 'PPa', 'PJ', 'PW', 'PC', 'PV', 'PF', 'Pohm', 'PS', 'PWb', 'PT', 'PH', 'Plm', 'Plx', 'PBq', 'PGy', 'PSv', 'Pkat', 'THz', 'TN', 'TPa', 'TJ', 'TW', 'TC', 'TV', 'TF', 'Tohm', 'TS', 'TWb', 'TT', 'TH', 'Tlm', 'Tlx', 'TBq', 'TGy', 'TSv', 'Tkat', 'GHz', 'GN', 'GPa', 'GJ', 'GW', 'GC', 'GV', 'GF', 'Gohm', 'GS', 'GWb', 'GT', 'GH', 'Glm', 'Glx', 'GBq', 'GGy', 'GSv', 'Gkat', 'MHz', 'MN', 'MPa', 'MJ', 'MW', 'MC', 'MV', 'MF', 'Mohm', 'MS', 'MWb', 'MT', 'MH', 'Mlm', 'Mlx', 'MBq', 'MGy', 'MSv', 'Mkat', 'kHz', 'kN', 'kPa', 'kJ', 'kW', 'kC', 'kV', 'kF', 'kohm', 'kS', 'kWb', 'kT', 'kH', 'klm', 'klx', 'kBq', 'kGy', 'kSv', 'kkat', 'hHz', 'hN', 'hPa', 'hJ', 'hW', 'hC', 'hV', 'hF', 'hohm', 'hS', 'hWb', 'hT', 'hH', 'hlm', 'hlx', 'hBq', 'hGy', 'hSv', 'hkat', 'daHz', 'daN', 'daPa', 'daJ', 'daW', 'daC', 'daV', 'daF', 'daohm', 'daS', 'daWb', 'daT', 'daH', 'dalm', 'dalx', 'daBq', 'daGy', 'daSv', 'dakat', 'dHz', 'dN', 'dPa', 'dJ', 'dW', 'dC', 'dV', 'dF', 'dohm', 'dS', 'dWb', 'dT', 'dH', 'dlm', 'dlx', 'dBq', 'dGy', 'dSv', 'dkat', 'cHz', 'cN', 'cPa', 'cJ', 'cW', 'cC', 'cV', 'cF', 'cohm', 'cS', 'cWb', 'cT', 'cH', 'clm', 'clx', 'cBq', 'cGy', 'cSv', 'ckat', 'mHz', 'mN', 'mPa', 'mJ', 'mW', 'mC', 'mV', 'mF', 'mohm', 'mS', 'mWb', 'mT', 'mH', 'mlm', 'mlx', 'mBq', 'mGy', 'mSv', 'mkat', 'muHz', 'muN', 'muPa', 'muJ', 'muW', 'muC', 'muV', 'muF', 'muohm', 'muS', 'muWb', 'muT', 'muH', 'mulm', 'mulx', 'muBq', 'muGy', 'muSv', 'mukat', 'nHz', 'nN', 'nPa', 'nJ', 'nW', 'nC', 'nV', 'nF', 'nohm', 'nS', 'nWb', 'nT', 'nH', 'nlm', 'nlx', 'nBq', 'nGy', 'nSv', 'nkat', 'pHz', 'pN', 'pPa', 'pJ', 'pW', 'pC', 'pV', 'pF', 'pohm', 'pS', 'pWb', 'pT', 'pH', 'plm', 'plx', 'pBq', 'pGy', 'pSv', 'pkat', 'fHz', 'fN', 'fPa', 'fJ', 'fW', 'fC', 'fV', 'fF', 'fohm', 'fS', 'fWb', 'fT', 'fH', 'flm', 'flx', 'fBq', 'fGy', 'fSv', 'fkat', 'aHz', 'aN', 'aPa', 'aJ', 'aW', 'aC', 'aV', 'aF', 'aohm', 'aS', 'aWb', 'aT', 'aH', 'alm', 'alx', 'aBq', 'aGy', 'aSv', 'akat', 'zHz', 'zN', 'zPa', 'zJ', 'zW', 'zC', 'zV', 'zF', 'zohm', 'zS', 'zWb', 'zT', 'zH', 'zlm', 'zlx', 'zBq', 'zGy', 'zSv', 'zkat', 'yHz', 'yN', 'yPa', 'yJ', 'yW', 'yC', 'yV', 'yF', 'yohm', 'yS', 'yWb', 'yT', 'yH', 'ylm', 'ylx', 'yBq', 'yGy', 'ySv', 'ykat']
```

Here for the imperial units : 

```python
from physipy import imperial_units
list(imperial_units.keys())
['in',
 'ft',
 'yd',
 'mi',
 'mil',
 'NM',
 'fur',
 'ac',
 'gallon',
 'quart',
 'pint',
 'cup',
 'foz',
 'tbsp',
 'tsp',
 'oz',
 'lb',
 'st',
 'ton',
 'slug',
 'kn',
 'lbf',
 'kip',
 'BTU',
 'cal',
 'kcal',
 'psi',
 'hp']
```

Finally, a dict containing all of those is available and will most frequently used : 

```python
from physipy import units
list(units.keys())
['m', 'kg', 's', 'A', 'K', 'mol', 'cd', 'rad', 'sr', 'Ym', 'Ykg', 'Ys', 'YA', 'YK', 'Ymol', 'Ycd', 'Yrad', 'Ysr', 'Zm', 'Zkg', 'Zs', 'ZA', 'ZK', 'Zmol', 'Zcd', 'Zrad', 'Zsr', 'Em', 'Ekg', 'Es', 'EA', 'EK', 'Emol', 'Ecd', 'Erad', 'Esr', 'Pm', 'Pkg', 'Ps', 'PA', 'PK', 'Pmol', 'Pcd', 'Prad', 'Psr', 'Tm', 'Tkg', 'Ts', 'TA', 'TK', 'Tmol', 'Tcd', 'Trad', 'Tsr', 'Gm', 'Gkg', 'Gs', 'GA', 'GK', 'Gmol', 'Gcd', 'Grad', 'Gsr', 'Mm', 'Mkg', 'Ms', 'MA', 'MK', 'Mmol', 'Mcd', 'Mrad', 'Msr', 'km', 'kkg', 'ks', 'kA', 'kK', 'kmol', 'kcd', 'krad', 'ksr', 'hm', 'hkg', 'hs', 'hA', 'hK', 'hmol', 'hcd', 'hrad', 'hsr', 'dam', 'dakg', 'das', 'daA', 'daK', 'damol', 'dacd', 'darad', 'dasr', 'dm', 'dkg', 'ds', 'dA', 'dK', 'dmol', 'dcd', 'drad', 'dsr', 'cm', 'ckg', 'cs', 'cA', 'cK', 'cmol', 'ccd', 'crad', 'csr', 'mm', 'g', 'ms', 'mA', 'mK', 'mmol', 'mcd', 'mrad', 'msr', 'mum', 'mukg', 'mus', 'muA', 'muK', 'mumol', 'mucd', 'murad', 'musr', 'nm', 'nkg', 'ns', 'nA', 'nK', 'nmol', 'ncd', 'nrad', 'nsr', 'pm', 'pkg', 'ps', 'pA', 'pK', 'pmol', 'pcd', 'prad', 'psr', 'fm', 'fkg', 'fs', 'fA', 'fK', 'fmol', 'fcd', 'frad', 'fsr', 'am', 'akg', 'as', 'aA', 'aK', 'amol', 'acd', 'arad', 'asr', 'zm', 'zkg', 'zs', 'zA', 'zK', 'zmol', 'zcd', 'zrad', 'zsr', 'ym', 'ykg', 'ys', 'yA', 'yK', 'ymol', 'ycd', 'yrad', 'ysr', 'Hz', 'N', 'Pa', 'J', 'W', 'C', 'V', 'F', 'ohm', 'S', 'Wb', 'T', 'H', 'lm', 'lx', 'Bq', 'Gy', 'Sv', 'kat', 'min', 'h', 'd', 'au', 'deg', 'ha', 'L', 't', 'Da', 'eV', 'YHz', 'YN', 'YPa', 'YJ', 'YW', 'YC', 'YV', 'YF', 'Yohm', 'YS', 'YWb', 'YT', 'YH', 'Ylm', 'Ylx', 'YBq', 'YGy', 'YSv', 'Ykat', 'ZHz', 'ZN', 'ZPa', 'ZJ', 'ZW', 'ZC', 'ZV', 'ZF', 'Zohm', 'ZS', 'ZWb', 'ZT', 'ZH', 'Zlm', 'Zlx', 'ZBq', 'ZGy', 'ZSv', 'Zkat', 'EHz', 'EN', 'EPa', 'EJ', 'EW', 'EC', 'EV', 'EF', 'Eohm', 'ES', 'EWb', 'ET', 'EH', 'Elm', 'Elx', 'EBq', 'EGy', 'ESv', 'Ekat', 'PHz', 'PN', 'PPa', 'PJ', 'PW', 'PC', 'PV', 'PF', 'Pohm', 'PS', 'PWb', 'PT', 'PH', 'Plm', 'Plx', 'PBq', 'PGy', 'PSv', 'Pkat', 'THz', 'TN', 'TPa', 'TJ', 'TW', 'TC', 'TV', 'TF', 'Tohm', 'TS', 'TWb', 'TT', 'TH', 'Tlm', 'Tlx', 'TBq', 'TGy', 'TSv', 'Tkat', 'GHz', 'GN', 'GPa', 'GJ', 'GW', 'GC', 'GV', 'GF', 'Gohm', 'GS', 'GWb', 'GT', 'GH', 'Glm', 'Glx', 'GBq', 'GGy', 'GSv', 'Gkat', 'MHz', 'MN', 'MPa', 'MJ', 'MW', 'MC', 'MV', 'MF', 'Mohm', 'MS', 'MWb', 'MT', 'MH', 'Mlm', 'Mlx', 'MBq', 'MGy', 'MSv', 'Mkat', 'kHz', 'kN', 'kPa', 'kJ', 'kW', 'kC', 'kV', 'kF', 'kohm', 'kS', 'kWb', 'kT', 'kH', 'klm', 'klx', 'kBq', 'kGy', 'kSv', 'kkat', 'hHz', 'hN', 'hPa', 'hJ', 'hW', 'hC', 'hV', 'hF', 'hohm', 'hS', 'hWb', 'hT', 'hH', 'hlm', 'hlx', 'hBq', 'hGy', 'hSv', 'hkat', 'daHz', 'daN', 'daPa', 'daJ', 'daW', 'daC', 'daV', 'daF', 'daohm', 'daS', 'daWb', 'daT', 'daH', 'dalm', 'dalx', 'daBq', 'daGy', 'daSv', 'dakat', 'dHz', 'dN', 'dPa', 'dJ', 'dW', 'dC', 'dV', 'dF', 'dohm', 'dS', 'dWb', 'dT', 'dH', 'dlm', 'dlx', 'dBq', 'dGy', 'dSv', 'dkat', 'cHz', 'cN', 'cPa', 'cJ', 'cW', 'cC', 'cV', 'cF', 'cohm', 'cS', 'cWb', 'cT', 'cH', 'clm', 'clx', 'cBq', 'cGy', 'cSv', 'ckat', 'mHz', 'mN', 'mPa', 'mJ', 'mW', 'mC', 'mV', 'mF', 'mohm', 'mS', 'mWb', 'mT', 'mH', 'mlm', 'mlx', 'mBq', 'mGy', 'mSv', 'mkat', 'muHz', 'muN', 'muPa', 'muJ', 'muW', 'muC', 'muV', 'muF', 'muohm', 'muS', 'muWb', 'muT', 'muH', 'mulm', 'mulx', 'muBq', 'muGy', 'muSv', 'mukat', 'nHz', 'nN', 'nPa', 'nJ', 'nW', 'nC', 'nV', 'nF', 'nohm', 'nS', 'nWb', 'nT', 'nH', 'nlm', 'nlx', 'nBq', 'nGy', 'nSv', 'nkat', 'pHz', 'pN', 'pPa', 'pJ', 'pW', 'pC', 'pV', 'pF', 'pohm', 'pS', 'pWb', 'pT', 'pH', 'plm', 'plx', 'pBq', 'pGy', 'pSv', 'pkat', 'fHz', 'fN', 'fPa', 'fJ', 'fW', 'fC', 'fV', 'fF', 'fohm', 'fS', 'fWb', 'fT', 'fH', 'flm', 'flx', 'fBq', 'fGy', 'fSv', 'fkat', 'aHz', 'aN', 'aPa', 'aJ', 'aW', 'aC', 'aV', 'aF', 'aohm', 'aS', 'aWb', 'aT', 'aH', 'alm', 'alx', 'aBq', 'aGy', 'aSv', 'akat', 'zHz', 'zN', 'zPa', 'zJ', 'zW', 'zC', 'zV', 'zF', 'zohm', 'zS', 'zWb', 'zT', 'zH', 'zlm', 'zlx', 'zBq', 'zGy', 'zSv', 'zkat', 'yHz', 'yN', 'yPa', 'yJ', 'yW', 'yC', 'yV', 'yF', 'yohm', 'yS', 'yWb', 'yT', 'yH', 'ylm', 'ylx', 'yBq', 'yGy', 'ySv', 'ykat']
```

#### Constants

Constants do not differ much from quantities and units : they are just `Quantity` with a specific value and a specific `Dimension`.
The `scipy.constants` modules exposes lots of constants' values, that are wrapped in quantities with their associated dimension, and available in a `constants` dict : 

```python
from physipy import constants

list(constants.keys())
['c',
 'speed_of_light',
 'mu_0',
 'epsilon_0',
 'h',
 'Planck',
 'hbar',
 'G',
 'gravitational_constant',
 'g',
 'e',
 'elementary_charge',
 'R',
 'gas_constant',
 'alpha',
 'fine_structure',
 'N_A',
 'Avogadro',
 'k',
 'Boltzmann',
 'sigma',
 'Stefan_Boltzmann',
 'Wien',
 'Rydberg',
 'm_e',
 'electron_mass',
 'm_p',
 'proton_mass',
 'm_n',
 'neutron_mass',
 'yotta',
 'zetta',
 'exa',
 'peta',
 'tera',
 'giga',
 'mega',
 'kilo',
 'hecto',
 'deka ',
 'deci ',
 'centi',
 'milli',
 'micro',
 'nano ',
 'pico ',
 'femto',
 'atto ',
 'zepto',
 'kibi',
 'mebi',
 'gibi',
 'tebi',
 'pebi',
 'exbi',
 'zebi',
 'yobi',
 'gram',
 'metric_ton',
 'grain',
 'lb',
 'pound',
 'blob',
 'slinch',
 'slug',
 'oz',
 'ounce',
 'stone',
 'long_ton',
 'short_ton',
 'troy_ounce',
 'troy_pound',
 'carat',
 'm_u',
 'u',
 'atomic_mass',
 'deg',
 'arcmin',
 'arcminute',
 'arcsec',
 'arcsecond',
 'minute',
 'hour',
 'day',
 'week',
 'year',
 'Julian_year',
 'inch',
 'foot',
 'yard',
 'mile',
 'mil',
 'pt',
 'point',
 'survey_foot',
 'survey_mile',
 'nautical_mile',
 'fermi',
 'angstrom',
 'micron',
 'au',
 'astronomical_unit',
 'light_year',
 'parsec',
 'atm',
 'atmosphere',
 'bar',
 'torr',
 'mmHg',
 'psi',
 'hectare',
 'acre',
 'liter',
 'litre',
 'gallon',
 'gallon_US',
 'gallon_imp',
 'fluid_ounce',
 'fluid_ounce_US',
 'fluid_ounce_imp',
 'bbl',
 'barrel',
 'kmh',
 'mph',
 'mach',
 'speed_of_sound',
 'knot',
 'zero_Celsius',
 'degree_Fahrenheit',
 'eV',
 'electron_volt',
 'calorie',
 'calorie_th',
 'calorie_IT',
 'erg',
 'Btu',
 'Btu_IT',
 'Btu_th',
 'ton_TNT',
 'hp',
 'horsepower',
 'dyn',
 'dyne',
 'lbf',
 'pound_force',
 'kgf',
 'kilogram_force']
```


### Numpy compatibility

Numpy is very well supported, thanks to the proposed interfaces exposed by numpy and the fact that `Quantity` is (almost) agnostic to the type of its value : it can be a float just as much as an array.

To define a Quantity with a numpy.ndarray value you can use again both approaches (constructor and multiplication), while you'll most likely just use multiplication :

```python
# using constructor approach
time_sampling = physipy.Quantity([0.1, 0.2, 0.3], physipy.Dimension("T"))
print(time_sampling) # [0.1 0.2 0.3] s

# using multiplication
position_sampling = np.array([1,2,3]) * m
print(position_sampling) # [1 2 3] m
```

You can then play with those as you would with regular ndarrays, as long as you respect dimensional analysis :

```python
print(position_sampling / time_sampling)    # [10. 10. 10.] m/s
print(2 * position_sampling)                # [2 4 6] m
```

You still cannot add carrots and potatoes : 

```python
try:
    position_sampling + time_sampling
except Exception as e:
    print("You can't add a length and a time dummy !")
    print(e)
# You can't add a length and a time dummy !
# Dimension error : dimensions of operands are L and T, and are differents (length vs time).
```

Dimensional analysis still applies for functions :

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

# [ 1.000000e+00  6.123234e-17 -1.000000e+00]
# [ 1.000000e+00  6.123234e-17 -1.000000e+00]
```

For more information on numpy's support see [here](scientific-stack/numpy-support.md).

### Matplotlib compatibility

Using just one additionnal line of code, you can make matplotlib aware of physipy's units, and automaticaly add labels to the axes.

```python
import matplotlib.pyplot as plt
from physipy import setup_matplotlib, units, kg, s
km = units['km']

# this is the line that makes matplotlib aware of physipy
setup_matplotlib()

# we'll do mercury, venus, mars and the earth
masses = [
    3.301 * 10**31, # mercury's mass
    4.867 * 10**24, # venus'
    5.972 * 10**31, # mars'
    6.441 * 10**23, # earth
] * kg

# same order as for the masses
speeds =[47.9, 35, 29.8, 24.1] * km/s

fig, ax = plt.subplots()
ax.plot(masses, speeds, 'o')
```

For more information on matplotlib's support see [here](scientific-stack/matplotlib-support.md).

### Pandas compatibility

You can also make pandas handle physipy quantities almost transparently using [`physipandas`](https://github.com/mocquin/physipandas), which is another package that extends physipy capabilities to pandas.