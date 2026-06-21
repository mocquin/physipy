# API reference

This section documents the public API of physipy. The pages are generated
directly from the source-code docstrings (via [mkdocstrings](https://mkdocstrings.github.io/)),
so they always reflect the installed version.

- [Dimension](dimension-api.md) — the `Dimension` class, representing the
  physical dimensions of a quantity.
- [Quantity](quantity-api.md) — the core `Quantity` class pairing a numerical
  value with a `Dimension`.
- [Units](units-api.md) — the predefined physical units (`physipy.units`).
- [Constants](constants-api.md) — the predefined physical constants
  (`physipy.constants`).
- [Plotting](plotting-api.md) — the matplotlib integration for plotting
  quantities with automatic unit handling.
