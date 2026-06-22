This page explains the one contract that underpins every numeric use of
physipy: **how a quantity stores its value, and how to get a plain number back
out in the unit you want.** Getting this right avoids a whole class of
silent unit-scale bugs.

## The storage invariant: values are always SI

A `Quantity` stores its magnitude in **SI base units**, internally, regardless
of how it is displayed. Construction unit, arithmetic, favourite unit — none of
them change the stored value; they only change the dimension (where applicable)
or the display.

```python
from physipy import m, units
mm = units["mm"]

length = 5 * m
print(length.value)          # 5.0   -> SI (metres)

same = 5000 * mm
print(same.value)            # 5.0   -> still SI; mm was converted on construction
```

`.value` is therefore the number you want whenever you hand a quantity to
plain maths, numpy, scipy, plotting coordinates computed by hand, etc.

## Getting a plain number back

There are exactly two things you usually want, and a clean way to get each:

| You want… | Use | Result |
|---|---|---|
| the magnitude in **SI** base units | `q.value` | `5.0` for `5 * m` |
| the magnitude **expressed in unit `U`** | `q / U` | `5000.0` for `(5*m) / mm` |

```python
length = 5 * m
print(length.value)          # 5.0     (SI, metres)
print(length / mm)           # 5000.0  (the value in mm)
```

The second form works because dividing by a unit of the **same dimension**
gives a dimensionless result — and physipy returns a dimensionless result as a
**bare number** (a `float`/`ndarray`), not a `Quantity`:

```python
print(type(length / mm).__name__)   # float  -- not Quantity!
```

This demotion is convenient but worth remembering: once you divide out the
unit, there is no `.value`/`.dimension` to call anymore — you already have the
number.

## Display is separate from value: `favunit`, `to`, `into`

The "favourite unit" (`favunit`) and the `to`/`into`/`set_favunit` helpers
control **only how a quantity prints**. They never change `.value`.

```python
shown_in_mm = (5 * m).into(mm)     # display-only change
print(shown_in_mm)                 # 5000.0 mm   (pretty display)
print(shown_in_mm.value)           # 5.0         (.value is STILL SI!)
```

This is the trap to internalise: **`q.into(mm).value` is the SI magnitude
(`5.0`), not the value in mm (`5000.0`).** If you want the number in mm, divide
(`q / mm`); `into`/`to` are for display.

- `to(U)` — display in `U` (any dimension allowed).
- `into(U)` — same, but raises if `U` is dimensionally incompatible.
- `set_favunit(U)` / `ito(U)` / `iinto(U)` — in-place variants returning `self`.

## A common footgun: `_compute_value()` is display-only

`Quantity` has an internal helper `_compute_value()` that returns the value
**expressed in the favourite unit** (`self / favunit`), falling back to `.value`
when no favunit is set. It exists for formatting/plotting.

It is **not** an SI accessor, and using it in numeric code is a silent bug
waiting to happen. When a favunit with a non-unit *scale* is set, it differs
from `.value` by that scale factor:

```python
from physipy import K, units
mum = units["mum"]

q = 3 * units["W"] / units["m"]**2
q.favunit = 1e6 * units["W"] / units["m"]**2   # a favunit whose scale is 1e6

print(q.value)             # 3.0     -> SI
print(q._compute_value())  # 3.0e-6  -> value in the favunit (SI / 1e6)
```

If you then integrate, average, or otherwise combine `_compute_value()` output
with SI-valued data, you introduce a `1/scale` error that *cancels in ratios*
and *looks fine in favunit-consistent plots* — so it can go unnoticed for a
long time.

**Rule of thumb:** use `.value` for computation, `q / U` for "value in unit U",
and reserve `_compute_value()` for display/formatting code.

## Summary

- Stored value is **always SI**: `q.value`.
- Value **in unit `U`**: `q / U` (returns a bare number; the unit divides out).
- `favunit` / `to` / `into` change **display only** — `.value` stays SI, and
  `q.into(U).value` is *not* the value in `U`.
- Dimensionless results are returned as plain numbers, not `Quantity`.
- `_compute_value()` is a favunit-relative display helper, **not** an SI
  accessor — keep it out of numeric paths.
