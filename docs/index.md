# physipy

**physipy** lets you manipulate *physical quantities* in Python — the association
of a value (a scalar, a `numpy.ndarray`, and more) with a *physical unit* (such as
meter or joule). Dimensional consistency is enforced automatically.

```python
from physipy import units, constants

nm = units["nm"]      # nanometer
hp = constants["h"]   # Planck's constant
c  = constants["c"]   # speed of light

E_ph = hp * c / (500 * nm)   # energy of a 500 nm photon
print(E_ph)                  # 3.9728916483435158e-19 kg*m**2/s**2

E_ph.favunit = units["J"]    # choose a display ("favourite") unit
print(E_ph)                  # 3.9728916483435158e-19 J
```

## Where to go next

- **[Installation](installation.md)** — install from PyPI, including the optional extras.
- **[Quickstart](quickstart.md)** — a guided tour of physipy by example.
- **Scientific stack** — integration with
  [numpy](scientific-stack/numpy-support.md),
  [matplotlib](scientific-stack/matplotlib-support.md),
  [pandas](scientific-stack/pandas-support.md),
  [scipy](scientific-stack/scipy-support.md), and the
  [standard math module](scientific-stack/math-support.md).
- **[API reference](API/api-reference.md)** — `Dimension`, `Quantity`, units, constants, plotting.
- **[Development](development-guide/index.md)** — contributing, benchmarking, and performance.

## At a glance

- **Light-weight** — two core classes, `Dimension` and `Quantity`.
- **numpy-friendly** — 150+ functions and ufuncs work transparently on quantities.
- **Lean by default** — only numpy is required; scipy, matplotlib and sympy are
  [optional extras](installation.md).
- **Typed** — ships inline type hints (PEP 561).

Source code and issues live on
[GitHub](https://github.com/mocquin/physipy/); releases are on
[PyPI](https://pypi.org/project/physipy/).
