# Tips and gotchas

A grab-bag of practical idioms for everyday use, and the sharp edges that most
often trip people up. The gotchas are not bugs — they follow from physipy's
design (a `Quantity` wraps a single SI value plus one `Dimension`) — but they can
surprise you if you don't know them.

## Tips

### Get a plain number out

Two idioms, depending on what you want:

```python
from physipy import m, units

q = 3 * units["mm"]
q.value        # 0.003  -> magnitude in SI (metres)
q / units["mm"]  # 3.0  -> magnitude expressed in a chosen unit
```

See [Getting a plain number back](values-units-and-display.md#getting-a-plain-number-back).

### Control how a quantity displays — `favunit`

The *value* and the *displayed unit* are independent. Set a favourite unit for
display without changing the stored value:

```python
q = 3000 * m
q.favunit = units["km"]
print(q)        # 3.0 km
q.value         # 3000.0  (unchanged, still SI)
```

See [Display is separate from value](values-units-and-display.md#display-is-separate-from-value-favunit-to-into).

### Build arrays of quantities — `asqarray`

`np.array([q1, q2])` does **not** give a dimensioned array. Use `asqarray`:

```python
from physipy import asqarray, m
asqarray([1 * m, 2 * m, 3 * m])   # a single Quantity wrapping array([1., 2., 3.]) * m
```

### Guard functions with dimensions — decorators

`physipy` ships decorators (in `physipy.utils`, re-exported at top level) to
enforce/strip dimensions at function boundaries:

| Decorator | What it does |
| --- | --- |
| `check_dimension(units_in, units_out)` | validate the dimensions of inputs and outputs |
| `dimension_and_favunit(inputs, outputs)` | validate dimensions *and* attach a favunit to outputs |
| `set_favunit(*favunits_out)` | attach favunits to outputs |
| `drop_dimension` | strip dimensions and pass the SI magnitudes to the wrapped function (handy to interface with code that only accepts plain numbers) |

### Units and constants are just objects

There is no registry to instantiate — units and constants are plain `Quantity`
objects looked up by name:

```python
from physipy import units, imperial_units, constants
units["mm"], imperial_units["mile"], constants["c"]   # constants pulls in scipy lazily
```

### Check what numpy is supported

physipy implements numpy support function-by-function. Query it at runtime:

```python
import physipy
print(physipy.numpy_coverage())          # summary of implemented / missing / n-a
physipy.supported_numpy_functions(names=True)
```

See the [numpy support page](scientific-stack/numpy-support.ipynb) for the full,
generated coverage report.

## Gotchas

### Values are always stored in SI

A quantity normalises to SI at construction, so `.value` is *not* the number you
typed if you used a non-SI unit:

```python
(5 * units["mm"]).value   # 0.005, not 5
```

See [The storage invariant](values-units-and-display.md#the-storage-invariant-values-are-always-si).

### `_favunit_value()` is display-only — don't compute with it

`_favunit_value()` returns the value *expressed in the favunit*, not SI. Using it
in numeric code introduces a silent `1/scale` error that cancels in ratios and
looks fine in plots. **Rule of thumb:** `.value` for computation, `q / U` for
"value in unit U", `_favunit_value()` only for display.
See [the footgun section](values-units-and-display.md#a-common-footgun-_favunit_value-is-display-only).

### Dimensionless results collapse to plain numbers

When an operation cancels all dimensions, you get a bare Python/numpy number, not
a `Quantity`:

```python
(3 * m) / (1 * m)   # 3.0  (a float, not a Quantity)
```

So don't expect `.value` / `.dimension` on the result of a ratio — it's already a
plain number.

### Comparisons return plain booleans (but still dimension-check)

```python
import numpy as np
(np.arange(3) * m) > (1.5 * m)   # array([False, False,  True]) — a bool ndarray
(3 * m) > (1 * s)                # raises DimensionError
```

The unit is enforced on the operands but stripped from the boolean result.

### Angles (`rad`, `sr`) are real dimensions

physipy treats plane angle and solid angle as base dimensions. This catches
real errors (mixing an angle with a bare number), but when you interface with
code that expects *dimensionless* radians you may need to drop the `rad`
dimension explicitly. Transcendental functions require dimensionless/angle
input:

```python
np.cos(3 * m)   # raises DimensionError (cosine of a length is meaningless)
```

### Some numpy functions can't be implemented

Because every element of a `Quantity` shares one `Dimension`, functions whose
output would need *per-element heterogeneous* dimensions have no faithful
representation (e.g. `np.vander`, the polynomial-coefficient family). See
[Functions that cannot be implemented](scientific-stack/numpy-support.ipynb#functions-that-cannot-be-implemented).

Two more numpy specifics:

- **Logical ufuncs** (`np.logical_and/or/xor/not`) are intentionally not
  implemented — their semantics on dimensioned values are ill-defined.
- **`np.arange` can't be overridden** the way ufuncs can. Use
  `physipy.quantity.utils.qarange`, or build from a plain range:

  ```python
  np.arange(10 * m)            # raises DimensionError
  from physipy.quantity.utils import qarange
  qarange(2 * m, 5 * m)        # array([2., 3., 4.]) * m
  np.arange(10) * m            # also fine
  ```

### matplotlib: activate the unit interface

By default matplotlib won't put units on your axes. Turn on the interface once
per session:

```python
import physipy
physipy.setup_matplotlib()   # now axis labels carry units automatically
```

See the [matplotlib support page](scientific-stack/matplotlib-support.ipynb).

### Limited quantity string parsing

physipy parses *dimension* strings (with the optional `sympy` extra) but not full
quantity strings like pint's `"3 m/s"`. See the
[comparison page limitations](comparison-with-other-packages.md#limitations-of-physipy).

## Reporting a gotcha

Known gotchas and limitations are tracked on GitHub with the **`gotcha`** label.

- Browse the current list:
  [open `gotcha` issues](https://github.com/mocquin/physipy/issues?q=is%3Aissue+label%3Agotcha).
- Hit a new one? [Open an issue](https://github.com/mocquin/physipy/issues/new)
  and tag it `gotcha` so it shows up in that list (and, eventually, here).
