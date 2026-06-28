# numpy integration: design & stability

This page explains *how* physipy plugs into numpy, *why* it is built that way,
and whether that approach is future-proof. For the concrete, always-up-to-date
list of *what* is supported, see the
[numpy support page](../scientific-stack/numpy-support.ipynb) and its generated
`physipy.numpy_coverage()` report.

## How physipy plugs into numpy

A `Quantity` is a **duck array**: a thin wrapper holding one ndarray of
magnitudes plus one shared `Dimension`. It is *not* a subclass of `ndarray`, and
units are *not* encoded in a custom dtype. Instead, physipy implements numpy's
two override protocols so that calling a numpy function on a `Quantity` dispatches
into unit-aware logic:

- **`__array_ufunc__`** ([NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html))
  handles ufuncs — element-wise operations (`np.add`, `np.sin`, `np.sqrt`, ...)
  and their `reduce` / `accumulate` / `out=` variants. physipy uses it to
  enforce/propagate dimensions (e.g. `sin` requires a dimensionless or angle
  input; `sqrt` halves the dimension exponents).
- **`__array_function__`** ([NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html))
  handles the rest of the high-level API (`np.concatenate`, `np.unique`,
  `np.linalg.norm`, the `np.fft` family, ...), function by function.

When neither protocol applies — for instance `np.asarray(quantity)` — physipy
falls back to converting to a plain ndarray, which **strips the unit and emits a
warning**. This is the deliberate escape hatch: leaving physipy's world is
allowed, but never silent.

## Is this approach future-proof?

**Yes.** For a library that *wraps* numpy arrays, the two override protocols are
the right, stable mechanism:

- **`__array_ufunc__` (NEP 13) is Final.** Ufuncs are the foundation of numpy;
  the protocol is a permanent part of the API.
- **`__array_function__` (NEP 18) is Final and enabled by default** since numpy
  1.17. It is **not** deprecated. (Older NEP text about "removing the checks in
  the next major release" refers to the historical 1.16 → 1.17 transition that
  dropped the opt-in environment variable, not to removing the protocol.)

There is one caveat worth knowing, stated by NEP 18 itself: while the *protocol*
is stable, its **use on any particular function** is considered experimental and
may change with little warning. In other words, churn — if any — lives in the
long tail of individual functions on a numpy upgrade, not in the dispatch
mechanism. physipy mitigates this with `physipy.numpy_coverage()` (which compares
the *running* numpy against what is implemented) plus the numpy test suite, so a
behavioural change in a specific function surfaces quickly.

### What about the Array API standard?

The [Python Array API standard](https://numpy.org/doc/stable/reference/array_api.html)
(entry point `__array_namespace__`) is sometimes assumed to supersede these
protocols. It does **not** — it is *complementary* and solves the opposite
problem:

- The **Array API standard** lets you write *portable consumer code* that runs
  unchanged across numpy, CuPy, JAX, PyTorch, etc.
- physipy's job is the **reverse**: make `np.func(quantity)` dispatch *into*
  unit-aware logic. That is exactly the duck-array override use case that NEP
  13 / NEP 18 were designed for.

numpy 2.0 removed the experimental `numpy.array_api` submodule and folded full
Array API support into the main namespace, so the two concerns now live side by
side. Adopting `__array_namespace__` would only matter if physipy wanted to be
*consumed as a backend* by Array-API-generic code — a different goal from making
numpy work on quantities.

## Why not subclass `ndarray` or use a custom dtype?

Two alternative architectures were considered and rejected:

- **Subclassing `ndarray`** (the route taken by `np.matrix`, `unyt`, and in part
  astropy) is more fragile — it pulls in `__array_finalize__` / `__array_wrap__`
  semantics — and is now generally discouraged in favour of duck-array wrappers
  like physipy's.
- **Encoding units in a custom dtype** (numpy's user-DType API) is a poor fit for
  physipy's data model: a dtype is *per element*, whereas a `Quantity` is *one
  array sharing a single* `Dimension`. The user-DType API is also still maturing.

The duck-array wrapper keeps the model simple ("one array + one dimension") and
matches numpy's current recommendation for array-like libraries.

## Directions

No change to the core approach is planned — it is the right one. The areas worth
keeping an eye on are:

- **Track numpy releases.** Run the suite against the supported numpy floor and
  the latest release; pair it with `numpy_coverage()` so a function whose
  dispatch changes upstream is caught early.
- **Keep the conversion escape hatch explicit and loud**, so dropping a unit on
  the way out to plain numpy is always an intentional, visible step.
- **`__array_namespace__` is optional and low priority** — only relevant if
  backend-agnostic (CuPy/JAX-backed) quantities ever become a goal.

## References

- [NEP 13 — A mechanism for overriding ufuncs](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
- [NEP 18 — A dispatch mechanism for numpy's high level array functions](https://numpy.org/neps/nep-0018-array-function-protocol.html)
- [NEP 56 — Array API standard support in numpy's main namespace](https://numpy.org/neps/nep-0056-array-api-main-namespace.html)
- [numpy: Array API standard compatibility](https://numpy.org/doc/stable/reference/array_api.html)
- [numpy support in physipy](../scientific-stack/numpy-support.ipynb) — the generated coverage report
