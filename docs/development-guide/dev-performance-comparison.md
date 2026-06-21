# Performance vs other packages

physipy is benchmarked against the other main physical-quantity libraries —
[pint](https://pint.readthedocs.io/), [astropy.units](https://docs.astropy.org/en/stable/units/)
and [forallpeople](https://github.com/connorferster/forallpeople) — plus a
unit-less **raw** float/`ndarray` baseline. The benchmark lives in the
repository at
[`benchmarks/compare_packages.py`](https://github.com/mocquin/physipy/blob/master/benchmarks/compare_packages.py)
and can be reproduced in one command.

## What is measured

The script compares the libraries on two axes:

- **Capability** — *which* operations each library actually supports (unary and
  binary operators, numpy ufuncs, and numpy functions), discovered by trying
  each operation and catching failures. This is a qualitative comparison, not a
  timing.
- **Speed** — time per call for a curated set of common operations, reported in
  microseconds and **relative to the raw-numpy baseline** (`×numpy`), for both
  scalar and array (length 1000) values.

Binary operations are exercised against three operand relationships (an idea
borrowed from [quantities-comparison](https://github.com/tbekolay/quantities-comparison)):

| Relationship  | Example          | What it exercises               |
| ------------- | ---------------- | ------------------------------- |
| `same`        | meter `+` meter  | plain arithmetic                |
| `compatible`  | meter `+` mile   | same dimension, unit conversion |
| `different`   | meter `+` second | different dimension             |

Each timing is the best of several `timeit` repeats, with the number of inner
loops chosen automatically.

## Reproduce

```bash
uv sync --group benchmark      # installs pint, astropy, forallpeople, ...
uv run python benchmarks/compare_packages.py
# options: --size N  --repeat R  --csv PATH  --plot PATH  --no-plot
```

It prints the tables below, writes a timings CSV and a capability CSV, and saves
the chart.

## Results

!!! note
    Microbenchmark on a single machine — read the numbers as *relative*
    comparisons (the `×numpy` column), not absolute throughput. Re-run locally
    for your own hardware and library versions.

### Capability (supported / total)

How many operations of each category each library accepts on array operands:

```text
library            unary_op     binary_op   unary_ufunc  binary_ufunc    numpy_func
raw               3/3          36/36         18/18         12/12         11/11
physipy           3/3          25/36         11/18         11/12         11/11
pint              3/3          25/36         11/18         11/12         11/11
astropy           3/3          25/36         11/18         11/12         11/11
forallpeople      2/3          11/36          8/18          8/12         10/11
```

physipy, pint and astropy share an essentially identical capability profile.
The "missing" unary ufuncs are transcendental functions such as `exp`, `log`
and `sin` applied to a *length*: these are correctly **rejected** as
dimensionally invalid (a feature, not a gap). forallpeople, being
scalar-oriented, supports far fewer operations.

### Timing — arrays (length 1000)

Time per call, `µs` and `×numpy` overhead:

```text
operation                      raw           physipy              pint           astropy      forallpeople
create                0.411 x    1      2.957 x    7      3.720 x    9      2.605 x    6    408.723 x  994
neg                   0.280 x    1      0.869 x    3      1.480 x    5      2.084 x    7    418.799 x 1496
abs                   0.295 x    1      0.880 x    3      1.486 x    5      2.094 x    7     28.849 x   98
add.same              0.340 x    1      1.029 x    3      3.427 x   10      2.475 x    7    531.605 x 1563
add.compatible        0.333 x    1      1.048 x    3      8.752 x   26      3.719 x   11               N/A
sub                   0.328 x    1      1.022 x    3      3.364 x   10      2.436 x    7    530.913 x 1621
mul                   0.329 x    1      2.465 x    7      3.575 x   11      4.929 x   15   5933.254 x18007
div                   0.364 x    1      2.516 x    7      3.763 x   10      2.773 x    8   2658.620 x 7312
pow                   0.306 x    1      2.753 x    9      3.257 x   11      3.801 x   12   1477.430 x 4829
less_than             0.400 x    1      0.683 x    2      1.109 x    3      1.855 x    5     78.385 x  196
np.sqrt               0.447 x    1      2.823 x    6      5.197 x   12      5.206 x   12   2331.691 x 5214
np.sum                1.023 x    1      2.102 x    2     12.499 x   12      3.579 x    3    524.820 x  513
np.mean               1.422 x    1      2.473 x    2     17.663 x   12      3.503 x    2    536.508 x  377
np.max                0.971 x    1      1.954 x    2     16.857 x   17      3.523 x    4     81.995 x   84
np.sort              12.619 x    1     13.681 x    1     27.756 x    2     13.417 x    1    640.278 x   51
np.concatenate        0.542 x    1      2.040 x    4      8.661 x   16      3.506 x    6      3.869 x    7
```

### Timing — scalars

```text
operation                      raw           physipy              pint           astropy      forallpeople
create                0.034 x    1      2.402 x   71      3.003 x   89      2.593 x   77      0.446 x   13
neg                   0.032 x    1      0.569 x   18      1.064 x   33      2.121 x   67      0.449 x   14
add.same              0.036 x    1      0.657 x   18      2.840 x   79      2.572 x   71      0.572 x   16
add.compatible        0.035 x    1      0.652 x   19      6.649 x  189      3.300 x   94               N/A
mul                   0.036 x    1      2.023 x   56      2.995 x   83      5.025 x  139      5.645 x  156
np.sum                1.102 x    1      2.136 x    2     12.450 x   11      3.412 x    3      1.275 x    1
np.sqrt               0.263 x    1      2.528 x   10      4.968 x   19      4.977 x   19      2.765 x   10
```

### Chart

![Cross-package benchmark: scalar timing, array timing, and capability coverage](../ressources/compare_packages.png)

## Takeaways

- **physipy has the lowest overhead of the unit libraries on essentially every
  operation** — typically 2–7× numpy on arrays, where pint and astropy are
  often 2–4× higher.
- The `compatible` relation exposes **unit-conversion cost**: `add.compatible`
  (meter + mile) pushes pint from 10× to 26× numpy and astropy from 7× to 11×,
  while physipy stays at 3× because it normalises to SI at construction.
- **forallpeople is scalar-oriented**: on arrays it builds object arrays of
  scalar `Physical`s, so array operations are 100–18000× slower — fine for
  engineering scalars, unsuitable for array workloads.
- numpy reductions (`sum`, `mean`, `max`) are cheap in physipy and astropy
  (2–4× numpy) but markedly slower in pint (12–17×).

See also the [airspeed-velocity benchmarks](dev-benchmarking-with-asv.md), which
track physipy's own performance over time, and the
[alternative packages](../misc/alternative-packages.md) page for a qualitative
comparison.
