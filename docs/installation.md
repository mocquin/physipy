# Installation

physipy requires **Python ≥ 3.10**. The core depends only on numpy; heavier
libraries are optional (see [Optional extras](#optional-extras) below).

## From PyPI (recommended)

The latest release is on [PyPI](https://pypi.org/project/physipy/):

```bash
pip install physipy
```

## Optional extras

Install only the features you need:

| Extra        | Enables                                                        | Pulls in     |
| ------------ | ------------------------------------------------------------- | ------------ |
| `calculus`   | `physipy.calculus` (integration / ODE / root finding)         | scipy        |
| `constants`  | physical-constant values in `physipy.constants`               | scipy        |
| `plotting`   | unit-aware matplotlib integration (`setup_matplotlib`)        | matplotlib   |
| `symbolic`   | compound dimension parsing (`Dimension("L**2/T")`) and LaTeX  | sympy        |
| `all`        | everything above                                              | all of them  |

```bash
pip install "physipy[plotting]"   # core + matplotlib integration
pip install "physipy[all]"        # everything
```

If you call an optional feature without its dependency installed, physipy raises
a clear `ImportError` telling you which extra to install.

## From source

The source code is hosted on
[GitHub](https://github.com/mocquin/physipy/). Clone and install in editable
mode (the `dev` group adds pytest, ruff, mypy and the full scientific stack):

```bash
git clone https://github.com/mocquin/physipy
cd physipy
pip install -e ".[all]"
```

If you use [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras        # core + extras + dev tooling
```

See the [development guide](development-guide/index.md) for the full contributor
workflow.
