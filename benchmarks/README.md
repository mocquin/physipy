# benchmarks

Cross-package comparison, separate from the
[airspeed-velocity](https://asv.readthedocs.io/) suite in `asv_benchmarks/`
(which tracks physipy's own performance over time).

[`compare_packages.py`](compare_packages.py) compares physipy with other
physical-quantity libraries — **pint**, **astropy.units**, **forallpeople** — and
a unit-less raw float/ndarray baseline, on two axes:

- **capability** — which operations each library actually supports (unary &
  binary operators, numpy ufuncs, numpy functions), discovered by trying them
  and catching failures;
- **speed** — time per call for a curated set of operations, in microseconds and
  relative to the raw-numpy baseline (`×numpy`).

Binary operations are exercised against three operand relationships (inspired by
[quantities-comparison](https://github.com/tbekolay/quantities-comparison)):

| Relationship  | Example          | What it exercises                  |
| ------------- | ---------------- | ---------------------------------- |
| `same`        | meter `+` meter  | plain arithmetic                   |
| `compatible`  | meter `+` mile   | same dimension, unit conversion    |
| `different`   | meter `+` second | different dimension                |

## Run

```bash
uv sync --group benchmark           # installs pint, astropy, forallpeople, ...
uv run python benchmarks/compare_packages.py
```

(or `pip install pint astropy forallpeople matplotlib pandas`, then
`python benchmarks/compare_packages.py`)

Options:

```
--size N        array length (default 1000)
--repeat R      timeit repeats, best-of (default 5)
--csv PATH      timings CSV (default compare_packages.csv; a sibling
                *_capability.csv is written alongside)
--plot PATH     chart: scalar timing, array timing, capability coverage
--no-plot       skip the chart
```

Missing libraries are skipped; unsupported operations are reported as `N/A`
(timing) or `0` (capability CSV).

## Outputs

- stdout: a capability table (supported/total per category) and a timing table
  (`µs [×numpy]`) for scalars and arrays;
- `compare_packages.csv` — raw per-operation timings;
- `compare_packages_capability.csv` — the full per-operation support matrix;
- `compare_packages.png` — grouped bar charts.
