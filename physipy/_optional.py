"""Helpers for optional third-party dependencies.

physipy's core (dimensional algebra on numpy values) only needs numpy. Some
features rely on heavier packages that are declared as optional extras in
pyproject.toml :

 - scipy       -> physipy.calculus and physipy.constants   (extra: ``calculus``)
 - matplotlib  -> the matplotlib unit integration / plots   (extra: ``plotting``)

Modules implementing those features import their dependency through
:func:`require`, so a minimal install (``pip install physipy``) stays lean and
only raises a clear, actionable error when an optional feature is actually
used.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(package: str, extra: str) -> ModuleType:
    """Import and return ``package``, or raise a helpful ImportError.

    Parameters
    ----------
    package : str
        The importable module name (e.g. ``"scipy.integrate"``).
    extra : str
        The physipy optional-dependency extra that provides it (e.g.
        ``"calculus"``), surfaced in the error message.
    """
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        top = package.split(".")[0]
        raise ImportError(
            f"'{top}' is required for this feature of physipy. Install it with "
            f"`pip install physipy[{extra}]` (or `pip install {top}`)."
        ) from exc
