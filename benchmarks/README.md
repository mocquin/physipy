# benchmarks

Cross-package performance comparison, separate from the
[airspeed-velocity](https://asv.readthedocs.io/) suite in `asv_benchmarks/`
(which tracks physipy's own performance over time).

[`compare_packages.py`](compare_packages.py) benchmarks physipy against other
physical-quantity libraries — **pint**, **astropy.units**, **forallpeople** — and
a unit-less raw float/ndarray baseline, on basic operations (creation, unary,
binary, comparison and numpy ufuncs) for both scalars and arrays.

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
--csv PATH      raw timings output (default compare_packages.csv)
--plot PATH     grouped bar chart (default compare_packages.png)
--no-plot       skip the chart
```

Missing libraries are skipped; operations a library doesn't support (e.g. numpy
ufuncs on forallpeople) are reported as `N/A`.
