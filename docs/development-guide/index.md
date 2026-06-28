# Development guide

physipy uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. All project metadata, dependencies, and tool configuration live in a
single [`pyproject.toml`](https://github.com/mocquin/physipy/blob/master/pyproject.toml).

## Setup

```bash
git clone https://github.com/mocquin/physipy
cd physipy
uv sync --all-extras            # core + optional extras + the `dev` group
pre-commit install              # run lint/format/type checks on commit
```

(With plain pip: `pip install -e ".[all]"` and `pip install --group dev`.)

## Testing

Run the test suite with pytest:

```bash
pytest
pytest --cov                    # with coverage (pytest-cov)
```

Doctests can be collected too:

```bash
pytest --doctest-modules physipy
```

## Linting, formatting and imports

[Ruff](https://docs.astral.sh/ruff/) handles linting, import sorting, and
formatting (configured under `[tool.ruff]` in `pyproject.toml`):

```bash
ruff check .                    # lint (add --fix to autofix)
ruff format .                   # format (add --check to verify only)
```

## Type checking

physipy ships inline type hints (PEP 561, via the `py.typed` marker). Check them
with [mypy](https://mypy-lang.org/) (configured under `[tool.mypy]`):

```bash
mypy
```

## Benchmarking

Performance is tracked with [asv](https://github.com/airspeed-velocity/asv);
results are published at [https://mocquin.github.io/physipy/](https://mocquin.github.io/physipy/).

[![asv benchmarks](./../ressources/asv_screenshot.png)](https://mocquin.github.io/physipy/)

See the [benchmarking with airspeed velocity](dev-benchmarking-with-asv.md) page
for details.

## Documentation

The docs are built with [mkdocs](https://www.mkdocs.org/),
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), and
[mkdocstrings](https://mkdocstrings.github.io/) (which pulls physipy's docstrings
into the API reference) and [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)
(which renders the scientific-stack notebooks). The configuration is in
`mkdocs.yml`; the docs dependencies are declared in the `docs` dependency group
of `pyproject.toml`.

```bash
uv sync --group docs            # install the docs toolchain
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build                    # build the static site into _mkdocks_site/
```

Most documentation pages live in [`docs/`](https://github.com/mocquin/physipy/tree/master/docs)
as Markdown. The scientific-stack pages are Jupyter notebooks
(`docs/scientific-stack/*.ipynb`) rendered directly by mkdocs-jupyter using the
outputs stored in the notebook (`execute: false`) — so re-run a notebook and
save it to refresh its rendered output and plots; there is no separate
Markdown-export step.

The site is hosted on [Read the Docs](https://physipy.readthedocs.io/); the build
is driven by `.readthedocs.yaml`, which installs the `docs` group with uv and runs
`mkdocs build`.

## Releasing

The version is the single value in
[`physipy/_version.py`](https://github.com/mocquin/physipy/blob/master/physipy/_version.py)
(read at build time by hatchling). Releases are tagged with the **bare version**
(`0.3.1`, no `v` prefix), matching the existing tag history.

A release is cut with one command —
[`scripts/release.py`](https://github.com/mocquin/physipy/blob/master/scripts/release.py),
which bumps the version, rolls `CHANGELOG.md`, builds, smoke-tests on TestPyPI,
publishes to PyPI, then commits, tags and pushes:

```bash
# preview everything first — prints each step, changes nothing
uv run python scripts/release.py patch --dry-run

# do it (patch | minor | major, or an explicit X.Y.Z)
uv run python scripts/release.py patch
```

Notes:

- **Add release notes as you go** under `## [Unreleased]` in
  [`CHANGELOG.md`](https://github.com/mocquin/physipy/blob/master/CHANGELOG.md);
  the script moves them under the new dated version automatically.
- Useful flags: `--no-testpypi` (skip the smoke upload), `--no-push` (commit and
  tag locally only), `--allow-dirty`, `-y` (non-interactive).
- **Consistency guard:** `uv run python scripts/release.py check` fails if the
  latest git tag does not match `__version__` (also accepts `--tag X.Y.Z`).
- PyPI / TestPyPI credentials are read by `uv publish` from your environment
  (e.g. `UV_PUBLISH_TOKEN`) or `~/.pypirc`; the script never handles secrets.

The pure logic (version bump, the tag/version guard, the changelog roll) is
unit-tested in [`test/test_release.py`](https://github.com/mocquin/physipy/blob/master/test/test_release.py).
