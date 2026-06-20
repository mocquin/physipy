"""Cross-package performance comparison for physical-quantity libraries.

Benchmarks physipy against pint, astropy.units and forallpeople (plus a
unit-less raw float/ndarray baseline) on basic operations - creation, unary,
binary, comparison and numpy ufuncs - for both scalar and array values.

Any library that is not installed is simply skipped; any operation a library
does not support is reported as N/A (this is common for forallpeople, which is
scalar-oriented).

Usage
-----
    # install the comparison libraries (see the `benchmark` dependency group):
    uv sync --group benchmark
    # or: pip install pint astropy forallpeople matplotlib pandas

    uv run python benchmarks/compare_packages.py
    uv run python benchmarks/compare_packages.py --size 10000 --repeat 7
    uv run python benchmarks/compare_packages.py --csv out.csv --plot out.png

Outputs a table to stdout, a CSV of raw timings, and (unless --no-plot) a
grouped bar chart of the timings.
"""

from __future__ import annotations

import argparse
import timeit
from collections import OrderedDict

import numpy as np


def build_libraries():
    """Return an ordered mapping name -> operand builders.

    Each entry exposes ``scalar()`` and ``array(n)`` callables returning a
    pair ``(x, y)`` of same-dimension operands (a length, in meters).
    Unavailable libraries are skipped.
    """
    libs = OrderedDict()

    def pair_scalar(unit):
        return lambda: (5.0 * unit, 3.0 * unit)

    def pair_array(unit):
        return lambda n: (
            np.linspace(1.0, 10.0, n) * unit,
            (np.linspace(1.0, 10.0, n) + 0.5) * unit,
        )

    # Unit-less baseline (meter == 1.0), to expose the per-op overhead.
    libs["raw"] = dict(scalar=pair_scalar(1.0), array=pair_array(1.0))

    try:
        from physipy import m

        libs["physipy"] = dict(scalar=pair_scalar(m), array=pair_array(m))
    except ImportError:
        pass

    try:
        import pint

        meter = pint.UnitRegistry().meter
        libs["pint"] = dict(scalar=pair_scalar(meter), array=pair_array(meter))
    except ImportError:
        pass

    try:
        from astropy import units as u

        libs["astropy"] = dict(scalar=pair_scalar(u.m), array=pair_array(u.m))
    except ImportError:
        pass

    try:
        import forallpeople as fp

        libs["forallpeople"] = dict(
            scalar=pair_scalar(fp.m), array=pair_array(fp.m)
        )
    except ImportError:
        pass

    return libs


# name -> callable(x, y). Binary ops use same-dimension operands; mul/div/pow
# deliberately change the dimension (a valid operation for every library).
OPERATIONS = OrderedDict(
    [
        ("create", lambda x, y: 5.0 * x),
        ("neg", lambda x, y: -x),
        ("abs", lambda x, y: abs(x)),
        ("add", lambda x, y: x + y),
        ("sub", lambda x, y: x - y),
        ("mul", lambda x, y: x * y),
        ("div", lambda x, y: x / y),
        ("pow", lambda x, y: x**2),
        ("less_than", lambda x, y: x < y),
        ("np.sqrt", lambda x, y: np.sqrt(x)),
        ("np.sum", lambda x, y: np.sum(x)),
        ("np.max", lambda x, y: np.max(x)),
    ]
)


def measure(func, repeat):
    """Best-of-``repeat`` time (seconds) for a single call of ``func``.

    Returns None if ``func`` raises (operation unsupported).
    """
    try:
        func()
    except Exception:
        return None
    timer = timeit.Timer(func)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeat, number=number)) / number


def run(libs, size, repeat):
    """Return results[scenario][op][lib] = seconds-per-call (or None)."""
    results = OrderedDict()
    for scenario in ("scalar", "array"):
        results[scenario] = OrderedDict(
            (op, OrderedDict()) for op in OPERATIONS
        )
        for name, lib in libs.items():
            try:
                x, y = (
                    lib["array"](size)
                    if scenario == "array"
                    else lib["scalar"]()
                )
            except Exception:
                x = y = None
            for op, fn in OPERATIONS.items():
                t = (
                    None
                    if x is None
                    else measure(lambda fn=fn, x=x, y=y: fn(x, y), repeat)
                )
                results[scenario][op][name] = t
    return results


def to_dataframe(scenario_results, libs):
    import pandas as pd

    # microseconds per call
    data = {
        op: [
            (scenario_results[op][lib] * 1e6)
            if scenario_results[op].get(lib) is not None
            else np.nan
            for lib in libs
        ]
        for op in scenario_results
    }
    return pd.DataFrame(data, index=list(libs)).T  # rows=ops, cols=libs


def print_report(results, libs):
    try:
        import pandas as pd  # noqa: F401

        have_pandas = True
    except ImportError:
        have_pandas = False

    for scenario in results:
        print(f"\n=== {scenario} : time per call (microseconds) ===")
        if have_pandas:
            df = to_dataframe(results[scenario], libs)
            with __import__("pandas").option_context(
                "display.float_format", lambda v: f"{v:9.3f}"
            ):
                print(df.to_string(na_rep="    N/A"))
        else:
            header = "op".ljust(12) + "".join(n.rjust(13) for n in libs)
            print(header)
            for op in results[scenario]:
                row = op.ljust(12)
                for lib in libs:
                    t = results[scenario][op].get(lib)
                    row += f"{t * 1e6:13.3f}" if t is not None else "      N/A"
                print(row)


def save_csv(results, libs, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "operation", *libs, "unit"])
        for scenario in results:
            for op in results[scenario]:
                row = [scenario, op]
                for lib in libs:
                    t = results[scenario][op].get(lib)
                    row.append("" if t is None else f"{t * 1e6:.6f}")
                row.append("microseconds")
                w.writerow(row)
    print(f"\nwrote {path}")


def save_plot(results, libs, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plot")
        return

    scenarios = list(results)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 9), squeeze=False)
    libnames = list(libs)
    for ax, scenario in zip(axes[:, 0], scenarios, strict=True):
        ops = list(results[scenario])
        xpos = np.arange(len(ops))
        width = 0.8 / max(len(libnames), 1)
        for i, lib in enumerate(libnames):
            heights = [
                (results[scenario][op].get(lib) or np.nan) * 1e6 for op in ops
            ]
            ax.bar(xpos + i * width, heights, width, label=lib)
        ax.set_yscale("log")
        ax.set_xticks(xpos + width * (len(libnames) - 1) / 2)
        ax.set_xticklabels(ops, rotation=30, ha="right")
        ax.set_ylabel("time per call (us, log)")
        ax.set_title(f"{scenario} operations")
        ax.legend(ncol=len(libnames), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--size", type=int, default=1000, help="array length (default 1000)"
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="timeit repeats (default 5)"
    )
    parser.add_argument(
        "--csv", default="compare_packages.csv", help="CSV output path"
    )
    parser.add_argument(
        "--plot", default="compare_packages.png", help="PNG output path"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip the chart"
    )
    args = parser.parse_args(argv)

    libs = build_libraries()
    print("libraries:", ", ".join(libs))
    print(f"array size: {args.size}, repeat: {args.repeat}")

    results = run(libs, args.size, args.repeat)
    print_report(results, libs)
    save_csv(results, libs, args.csv)
    if not args.no_plot:
        save_plot(results, libs, args.plot)


if __name__ == "__main__":
    main()
