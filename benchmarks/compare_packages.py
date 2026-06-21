"""Cross-package performance & capability comparison for unit libraries.

Compares physipy with pint, astropy.units and forallpeople (plus a unit-less
raw float/ndarray baseline) on two axes:

* **capability** - which operations each library actually supports (unary &
  binary operators, numpy ufuncs, and numpy functions), discovered by trying
  them and catching failures;
* **speed** - time per call for a curated set of common operations, reported
  both in microseconds and relative to the raw-numpy baseline (``xnumpy``).

Binary operations are exercised against three operand relationships, inspired
by Trevor Bekolay's quantities-comparison
(https://github.com/tbekolay/quantities-comparison):

* ``same``        - meter <op> meter        (plain arithmetic)
* ``compatible``  - meter <op> mile         (same dimension, unit conversion)
* ``different``   - meter <op> second       (different dimension)

Any library that is not installed is skipped; any unsupported operation is
reported as N/A.

Usage
-----
    uv sync --group benchmark      # pint, astropy, forallpeople, matplotlib...
    uv run python benchmarks/compare_packages.py
    uv run python benchmarks/compare_packages.py --size 10000 --repeat 7
    uv run python benchmarks/compare_packages.py --no-plot

Outputs: capability + timing tables to stdout, two CSVs, and a chart.
"""

from __future__ import annotations

import argparse
import operator as op
import timeit
from collections import OrderedDict

import numpy as np

# Operand relationship -> unit name passed to a library's ``make``.
RELATIONS = OrderedDict(
    [
        ("same", "meter"),  # meter <op> meter
        ("compatible", "mile"),  # meter <op> mile  (same dimension)
        ("different", "second"),  # meter <op> second (other dimension)
    ]
)


def build_libraries():
    """Return ``name -> make(value, unit_name)`` for the installed libraries.

    ``make`` raises (``KeyError``) for a unit the library doesn't provide, so
    the relevant operands are simply marked unavailable.
    """
    libs = OrderedDict()

    def adapter(unit_map):
        return lambda value, name: value * unit_map[name]

    # Unit-less baseline: ignore the unit entirely (no dimensional checking).
    libs["raw"] = lambda value, name: value * 1.0

    try:
        from physipy import imperial_units, units

        umap = {"meter": units["m"], "second": units["s"]}
        if "mi" in imperial_units:
            umap["mile"] = imperial_units["mi"]
        libs["physipy"] = adapter(umap)
    except ImportError:
        pass

    try:
        import pint

        ureg = pint.UnitRegistry()
        libs["pint"] = adapter(
            {"meter": ureg.meter, "mile": ureg.mile, "second": ureg.second}
        )
    except ImportError:
        pass

    try:
        from astropy import units as u

        libs["astropy"] = adapter(
            {"meter": u.m, "mile": u.imperial.mile, "second": u.s}
        )
    except ImportError:
        pass

    try:
        import forallpeople as fp

        umap = {"meter": fp.m}
        if hasattr(fp, "s"):
            umap["second"] = fp.s
        libs["forallpeople"] = adapter(umap)
    except ImportError:
        pass

    return libs


# --------------------------------------------------------------------------
# Operation catalogue
# --------------------------------------------------------------------------
UNARY_OPS = OrderedDict([("neg", op.neg), ("pos", op.pos), ("abs", op.abs)])

BINARY_OPS = OrderedDict(
    [
        ("add", op.add),
        ("sub", op.sub),
        ("mul", op.mul),
        ("truediv", op.truediv),
        ("floordiv", op.floordiv),
        ("mod", op.mod),
        ("pow", op.pow),
        ("lt", op.lt),
        ("le", op.le),
        ("eq", op.eq),
        ("ge", op.ge),
        ("gt", op.gt),
    ]
)

UNARY_UFUNCS = [
    np.negative,
    np.absolute,
    np.sqrt,
    np.square,
    np.reciprocal,
    np.rint,
    np.sign,
    np.floor,
    np.ceil,
    np.trunc,
    np.isfinite,
    np.exp,
    np.log,
    np.log10,
    np.sin,
    np.cos,
    np.arctan,
    np.deg2rad,
]

BINARY_UFUNCS = [
    np.add,
    np.subtract,
    np.multiply,
    np.true_divide,
    np.floor_divide,
    np.power,
    np.hypot,
    np.arctan2,
    np.maximum,
    np.minimum,
    np.greater,
    np.equal,
]

# name -> (callable(operands), required operand keys)
NUMPY_FUNCS = OrderedDict(
    [
        ("sum", (lambda o: np.sum(o["x"]), ["x"])),
        ("mean", (lambda o: np.mean(o["x"]), ["x"])),
        ("std", (lambda o: np.std(o["x"]), ["x"])),
        ("median", (lambda o: np.median(o["x"]), ["x"])),
        ("min", (lambda o: np.min(o["x"]), ["x"])),
        ("max", (lambda o: np.max(o["x"]), ["x"])),
        ("sort", (lambda o: np.sort(o["x"]), ["x"])),
        ("argsort", (lambda o: np.argsort(o["x"]), ["x"])),
        ("cumsum", (lambda o: np.cumsum(o["x"]), ["x"])),
        (
            "concatenate",
            (lambda o: np.concatenate([o["x"], o["same"]]), ["x", "same"]),
        ),
        ("where", (lambda o: np.where(o["x"] > o["same"]), ["x", "same"])),
    ]
)


def build_capability_ops():
    """Build ``(category, key, fn, required_keys)`` tuples for every op."""
    ops = []
    for name, fn in UNARY_OPS.items():
        ops.append(
            ("unary_op", name, (lambda f: lambda o: f(o["x"]))(fn), ["x"])
        )
    for name, fn in BINARY_OPS.items():
        for rel in RELATIONS:
            ops.append(
                (
                    "binary_op",
                    f"{name}.{rel}",
                    (lambda f, r: lambda o: f(o["x"], o[r]))(fn, rel),
                    ["x", rel],
                )
            )
    for uf in UNARY_UFUNCS:
        ops.append(
            (
                "unary_ufunc",
                uf.__name__,
                (lambda f: lambda o: f(o["x"]))(uf),
                ["x"],
            )
        )
    for bf in BINARY_UFUNCS:
        ops.append(
            (
                "binary_ufunc",
                bf.__name__,
                (lambda f: lambda o: f(o["x"], o["same"]))(bf),
                ["x", "same"],
            )
        )
    for name, (fn, required) in NUMPY_FUNCS.items():
        ops.append(("numpy_func", name, fn, required))
    return ops


# Curated operations to *time* (category, label, fn, required, array_only).
def build_timing_ops():
    return [
        ("create", "create", lambda o: 5.0 * o["x"], ["x"], False),
        ("unary", "neg", lambda o: -o["x"], ["x"], False),
        ("unary", "abs", lambda o: abs(o["x"]), ["x"], False),
        (
            "binary",
            "add.same",
            lambda o: o["x"] + o["same"],
            ["x", "same"],
            False,
        ),
        (
            "binary",
            "add.compatible",
            lambda o: o["x"] + o["compatible"],
            ["x", "compatible"],
            False,
        ),
        ("binary", "sub", lambda o: o["x"] - o["same"], ["x", "same"], False),
        ("binary", "mul", lambda o: o["x"] * o["same"], ["x", "same"], False),
        ("binary", "div", lambda o: o["x"] / o["same"], ["x", "same"], False),
        ("binary", "pow", lambda o: o["x"] ** 2, ["x"], False),
        (
            "binary",
            "less_than",
            lambda o: o["x"] < o["same"],
            ["x", "same"],
            False,
        ),
        ("ufunc", "np.sqrt", lambda o: np.sqrt(o["x"]), ["x"], False),
        ("numpy", "np.sum", lambda o: np.sum(o["x"]), ["x"], False),
        ("numpy", "np.mean", lambda o: np.mean(o["x"]), ["x"], False),
        ("numpy", "np.max", lambda o: np.max(o["x"]), ["x"], False),
        ("numpy", "np.sort", lambda o: np.sort(o["x"]), ["x"], True),
        (
            "numpy",
            "np.concatenate",
            lambda o: np.concatenate([o["x"], o["same"]]),
            ["x", "same"],
            True,
        ),
    ]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def make_operands(make, base_a, base_b):
    """Build the operand set for one library, marking missing units None."""
    out = {"x": _try_make(make, base_a, "meter")}
    out["same"] = _try_make(make, base_b, "meter")
    out["compatible"] = _try_make(make, base_b, "mile")
    out["different"] = _try_make(make, base_b, "second")
    return out


def _try_make(make, value, name):
    try:
        return make(value, name)
    except Exception:
        return None


def supported(fn, required, operands):
    if any(operands.get(k) is None for k in required):
        return False
    try:
        fn(operands)
    except Exception:
        return False
    return True


def measure(call, repeat):
    """Best-of-``repeat`` seconds per call (autoranged number of loops)."""
    timer = timeit.Timer(call)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeat, number=number)) / number


def base_values(scenario, size):
    if scenario == "scalar":
        return 5.0, 3.0
    arr = np.linspace(1.0, 10.0, size)
    return arr, arr + 0.5


def run(libs, size, repeat):
    cap_ops = build_capability_ops()
    timing_ops = build_timing_ops()
    capability = OrderedDict()  # cap[scenario][lib][key] = bool
    timing = OrderedDict()  # timing[scenario][label][lib] = seconds | None
    for scenario in ("scalar", "array"):
        a, b = base_values(scenario, size)
        capability[scenario] = OrderedDict()
        timing[scenario] = OrderedDict(
            (label, OrderedDict()) for _, label, *_ in timing_ops
        )
        for name, make in libs.items():
            operands = make_operands(make, a, b)
            capability[scenario][name] = OrderedDict(
                (key, supported(fn, req, operands))
                for _, key, fn, req in cap_ops
            )
            for _, label, fn, req, array_only in timing_ops:
                if array_only and scenario == "scalar":
                    timing[scenario][label][name] = None
                elif supported(fn, req, operands):
                    timing[scenario][label][name] = measure(
                        lambda fn=fn, operands=operands: fn(operands), repeat
                    )
                else:
                    timing[scenario][label][name] = None
    return capability, timing, cap_ops


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def categories_of(cap_ops):
    cats = OrderedDict()
    for cat, key, *_ in cap_ops:
        cats.setdefault(cat, []).append(key)
    return cats


def print_capability(capability, cap_ops, libs):
    cats = categories_of(cap_ops)
    for scenario in capability:
        print(f"\n=== {scenario} : capability (supported / total) ===")
        header = "library".ljust(13) + "".join(c.rjust(14) for c in cats)
        print(header)
        for lib in libs:
            row = lib.ljust(13)
            flags = capability[scenario][lib]
            for keys in cats.values():
                n = sum(flags[k] for k in keys)
                row += f"{n:>6}/{len(keys):<7}".rjust(14)
            print(row)


def print_timing(timing, libs):
    for scenario in timing:
        print(f"\n=== {scenario} : time per call (us) [xnumpy] ===")
        header = "operation".ljust(16) + "".join(n.rjust(18) for n in libs)
        print(header)
        for label in timing[scenario]:
            row = label.ljust(16)
            raw = timing[scenario][label].get("raw")
            for lib in libs:
                t = timing[scenario][label].get(lib)
                if t is None:
                    row += "N/A".rjust(18)
                else:
                    rel = f"x{t / raw:>5.0f}" if raw else "      "
                    row += f"{t * 1e6:9.3f} {rel}".rjust(18)
            print(row)


def save_timing_csv(timing, libs, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "operation", *libs, "unit"])
        for scenario in timing:
            for label in timing[scenario]:
                row = [scenario, label]
                for lib in libs:
                    t = timing[scenario][label].get(lib)
                    row.append("" if t is None else f"{t * 1e6:.6f}")
                row.append("microseconds")
                w.writerow(row)
    print(f"\nwrote {path}")


def save_capability_csv(capability, cap_ops, libs, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "category", "operation", *libs])
        for scenario in capability:
            for cat, key, *_ in cap_ops:
                row = [scenario, cat, key]
                for lib in libs:
                    row.append(int(capability[scenario][lib][key]))
                w.writerow(row)
    print(f"wrote {path}")


def save_plot(capability, timing, cap_ops, libs, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plot")
        return

    libnames = list(libs)
    cats = categories_of(cap_ops)
    scenarios = list(timing)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # timing, one subplot per scenario
    for ax, scenario in zip(axes[:2], scenarios, strict=True):
        labels = list(timing[scenario])
        xpos = np.arange(len(labels))
        width = 0.8 / max(len(libnames), 1)
        for i, lib in enumerate(libnames):
            heights = [
                (timing[scenario][lbl].get(lib) or np.nan) * 1e6
                for lbl in labels
            ]
            ax.bar(xpos + i * width, heights, width, label=lib)
        ax.set_yscale("log")
        ax.set_xticks(xpos + width * (len(libnames) - 1) / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("us per call (log)")
        ax.set_title(f"{scenario} - time per call")
        ax.legend(ncol=len(libnames), fontsize=8)

    # capability coverage (array scenario), grouped by category
    ax = axes[2]
    catnames = list(cats)
    xpos = np.arange(len(catnames))
    width = 0.8 / max(len(libnames), 1)
    for i, lib in enumerate(libnames):
        flags = capability["array"][lib]
        fracs = [
            100.0 * sum(flags[k] for k in cats[c]) / len(cats[c])
            for c in catnames
        ]
        ax.bar(xpos + i * width, fracs, width, label=lib)
    ax.set_xticks(xpos + width * (len(libnames) - 1) / 2)
    ax.set_xticklabels(catnames)
    ax.set_ylabel("% supported")
    ax.set_ylim(0, 105)
    ax.set_title("capability coverage (array operands)")
    ax.legend(ncol=len(libnames), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare unit libraries on speed and capability."
    )
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--csv", default="compare_packages.csv")
    parser.add_argument("--plot", default="compare_packages.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    libs = build_libraries()
    print("libraries:", ", ".join(libs))
    print(f"array size: {args.size}, repeat: {args.repeat}")

    capability, timing, cap_ops = run(libs, args.size, args.repeat)
    print_capability(capability, cap_ops, libs)
    print_timing(timing, libs)

    save_timing_csv(timing, libs, args.csv)
    cap_path = args.csv.replace(".csv", "_capability.csv")
    if cap_path == args.csv:
        cap_path = args.csv + ".capability.csv"
    save_capability_csv(capability, cap_ops, libs, cap_path)
    if not args.no_plot:
        save_plot(capability, timing, cap_ops, libs, args.plot)


if __name__ == "__main__":
    main()
