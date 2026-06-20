This page gives you a quick tour of physipy. For installation instructions, see
[here](installation.md).

## Quickstart by example

### Example 1: body-mass index

The simplest way to use physipy is to import some units and write your code with
them — which is why most of the time you'll write `from physipy import ...`
(rather than `import physipy`):

```python
from physipy import kg, m

# physical quantities that are unit-aware
my_weight = 75 * kg
my_height = 1.89 * m

def bmi(weight, height):
    "Compute Body-Mass-Index from weight and height."
    return weight / height**2

print(bmi(my_weight, my_height))
# --> 20.99605274208449 kg/m**2
```

You can express the same quantities in other units (centimeters, grams, …) and
get the same result, because physipy stores everything internally in SI units:

```python
from physipy import units

cm = units["cm"]   # stored as 1/100 of a meter
g  = units["g"]    # stored as 1/1000 of a kilogram

print(bmi(75_000 * g, 189 * cm))
# --> 20.99605274208449 kg/m**2
```

### Example 2: kinetic energy

Compute the kinetic energy $E_K = \tfrac{1}{2} m v^2$ of four planets, using
numpy-backed arrays of quantities:

```python
from physipy import units, s, kg

km = units["km"]
MJ = units["MJ"]

# mercury, venus, earth, mars
masses = [3.301e23, 4.867e24, 5.972e24, 6.417e23] * kg
speeds = [47.9, 35.0, 29.8, 24.1] * km / s   # mean orbital speeds

kinetic = 1 / 2 * masses * speeds**2
kinetic.favunit = MJ        # display in mega-joules
print(kinetic)
# [3.78692371e+26 2.98103750e+27 2.65168744e+27 1.86352889e+26] MJ
```

### Example 3: plotting Planck's law

This example combines three features: attaching units to numbers, the
`set_favunit` decorator (to set a display unit on a function's output), and the
matplotlib integration enabled by `setup_matplotlib()`:

```python
import numpy as np
import matplotlib.pyplot as plt

from physipy import units, constants, set_favunit, setup_matplotlib
from physipy import m, K, sr

W   = units["W"]
mum = units["mum"]
hp  = constants["Planck"]
c   = constants["c"]
kB  = constants["k"]

# attach a favourite (display) unit to the function output
@set_favunit(W / (m**2 * sr * mum))
def planck(wl, T):
    return 2 * hp * c**2 / wl**5 * 1 / (np.exp(hp * c / (wl * kB * T)) - 1) / sr

T_bb = 5800 * K                       # black-body temperature (a scalar)
wl = np.linspace(0.3, 3, 100) * mum   # wavelengths (an array)
wl.favunit = mum

setup_matplotlib()                    # make matplotlib unit-aware
plt.plot(wl, planck(wl, T_bb))        # axes are labelled with units automatically
```

## How physipy works

physipy is built on just two classes — `Dimension` and `Quantity`. Everything
else (units, constants, numpy/matplotlib support) is convenience built on top.

### Dimension

A `Dimension` represents a physical dimension as a mapping of base dimensions to
their exponents. You rarely create one directly, but it helps to see how it
works:

```python
import physipy

length = physipy.Dimension("L")              # length
speed  = physipy.Dimension({"L": 1, "T": -1})  # length / time
print(length)        # L
print(speed)         # L/T

print(length * speed)   # L**2/T
print(length / speed)   # T
```

A dimension can be rendered as its corresponding SI unit string:

```python
print(length.str_SI_unit())       # m
print(speed.str_SI_unit())        # m/s
print((length**2).str_SI_unit())  # m**2
print((1 / length).str_SI_unit()) # 1/m
```

The base dimensions are those of the SI system:

| Symbol  | Dimension                  | SI unit       |
| ------- | -------------------------- | ------------- |
| `L`     | length                     | meter (`m`)   |
| `M`     | mass                       | kilogram (`kg`) |
| `T`     | time                       | second (`s`)  |
| `I`     | electric current           | ampere (`A`)  |
| `theta` | thermodynamic temperature  | kelvin (`K`)  |
| `N`     | amount of substance        | mole (`mol`)  |
| `J`     | luminous intensity         | candela (`cd`) |
| `RAD`   | plane angle                | radian (`rad`) |
| `SR`    | solid angle                | steradian (`sr`) |

### Quantity

A `Quantity` is the association of a value (scalar, array, …) with a
`Dimension`. Create one by multiplying a number by a unit (most common), or via
the constructor:

```python
import physipy
from physipy import kg

mass = 2000 * kg                                  # by multiplication
mass = physipy.Quantity(2000, physipy.Dimension("M"))  # equivalent

print(mass.value)      # 2000
print(mass.dimension)  # M
```

Units themselves are just quantities with a value of 1:

```python
print(kg.value)      # 1
print(kg.dimension)  # M
```

Operations enforce dimensional analysis — compatible operands combine, and
incompatible ones raise `DimensionError`:

```python
from physipy import m, s

5 * m + 3 * m   # OK
5 * m + 3 * s   # raises DimensionError
```

### Favourite units

A quantity is displayed in SI units by default. Set its `favunit` (favourite
unit) — any quantity that carries a symbol — to change *only the display*; the
stored value and dimension are unchanged:

```python
from physipy import kg, units, constants

energy = 75 * kg * constants["c"]**2
print(energy)               # 6.740663840526132e+18 kg*m**2/s**2

energy.favunit = units["J"]
print(energy)               # 6.740663840526132e+18 J

energy.favunit = units["eV"]
print(energy)               # 4.2071914528533386e+37 eV
```

### Units and constants

Units and constants are plain `Quantity` objects exposed through dicts keyed by
name. `units` holds the SI units, their prefixed forms, derived units and a few
imperial units; `constants` wraps scipy's physical constants:

```python
from physipy import units, constants

units["mm"], units["N"], units["eV"]          # millimeter, newton, electron-volt
constants["c"], constants["h"], constants["k"] # speed of light, Planck, Boltzmann

# they are dicts, so explore them with the usual tools
sorted(units)[:5]       # ['A', 'Bq', 'C', 'Da', 'EA']  (sample)
len(units), len(constants)
```

physipy also exposes more specific dicts: `SI_units`, `SI_derived_units` and
`imperial_units`. See the [units API](API/units-api.md) and
[constants API](API/constants-api.md) for the full listings.

## Scientific-stack integration

physipy is designed to work transparently with the scientific Python stack:

- **[numpy](scientific-stack/numpy-support.md)** — 150+ functions and ufuncs work
  on quantities, with dimensional analysis applied throughout.
- **[matplotlib](scientific-stack/matplotlib-support.md)** — `setup_matplotlib()`
  labels axes with units automatically.
- **[scipy](scientific-stack/scipy-support.md)** and the
  **[standard math module](scientific-stack/math-support.md)**.
- **pandas** — via the companion package
  [`physipandas`](https://github.com/mocquin/physipandas).
