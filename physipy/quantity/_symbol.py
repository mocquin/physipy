"""A tiny pure-Python monomial used for the ``Quantity.symbol`` feature.

``Quantity.symbol`` carries a human-readable label (``m``, ``N*m``, ``m**2`` …)
that is combined through multiplication / division / exponentiation, with
canonical simplification (``m*m`` -> ``m**2``, ``m/m`` -> ``1``).

That is exactly a commutative *monomial* over symbol-name "atoms" with
integer/float exponents. It is **purely multiplicative**: addition and
subtraction of symbols are meaningless (``m + s`` has no unit interpretation)
and were never supported. They are explicitly rejected here, and an additive
sympy expression passed to :meth:`UnitSymbol.coerce` is rejected too, rather
than being silently flattened into an opaque atom.

physipy used to delegate this to sympy ; this class reproduces the needed
behaviour in pure Python, reusing the same canonical renderer as
:class:`~physipy.quantity.dimension.Dimension` (:func:`format_power_dict`), so
the displayed labels are byte-for-byte identical to the previous sympy output.

sympy is therefore no longer required ; it stays an optional dependency used
only for LaTeX rendering.
"""

from __future__ import annotations

from typing import Any, Union

from .dimension import Scalar, format_power_dict


def _normalize_exponent(exp: Any) -> Scalar:
    """Coerce an exponent to a plain python int (preferred) or float."""
    as_float = float(exp)
    as_int = int(as_float)
    return as_int if as_float == as_int else as_float


class UnitSymbol:
    """A product of ``name**exponent`` atoms with a canonical string form.

    Supported operations are strictly multiplicative: ``*`` (product), ``/``
    (quotient) and ``**`` (integer/float power), all of which keep the result
    a monomial. Addition and subtraction are **not** supported and raise
    :class:`TypeError` — there is no sensible "sum of units" label, and
    physipy never produces one (adding two quantities resets the result symbol
    to the default instead of combining symbols).
    """

    __slots__ = ("powers",)

    def __init__(
        self, powers: Union[str, dict, "UnitSymbol"]
    ) -> None:
        if isinstance(powers, str):
            # a single atom whose name is the whole string
            self.powers = {powers: 1}
        elif isinstance(powers, UnitSymbol):
            self.powers = dict(powers.powers)
        elif isinstance(powers, dict):
            # drop zero exponents so the canonical form / equality are stable
            self.powers = {k: v for k, v in powers.items() if v != 0}
        else:
            raise TypeError(
                f"UnitSymbol expects a str, dict or UnitSymbol, not "
                f"{type(powers)}"
            )

    @classmethod
    def coerce(cls, value: Any) -> "UnitSymbol":
        """Build a UnitSymbol from a str, dict, UnitSymbol, or sympy monomial.

        A *multiplicative* sympy expression (a product / quotient / power of
        symbols) is accepted by reading its ``as_powers_dict()``. An
        **additive** sympy expression (e.g. ``Symbol("a") + Symbol("b")``) is
        rejected with :class:`TypeError`: a sum of symbols is not a unit label
        and must not be silently turned into an opaque ``"a + b"`` atom. Any
        other type is rejected as well.
        """
        if isinstance(value, (cls, str, dict)):
            return cls(value)
        # duck-type sympy expressions (optional dependency) without importing
        # sympy : a monomial exposes as_powers_dict() -> {base: exponent}.
        as_powers_dict = getattr(value, "as_powers_dict", None)
        if callable(as_powers_dict):
            powers = {}
            for base, exp in as_powers_dict().items():
                # sympy tags an addition as `is_Add`; such a base means the
                # expression is not a monomial and is not a valid unit symbol.
                if getattr(base, "is_Add", False):
                    raise TypeError(
                        "additive symbol expressions are not supported : a "
                        f"unit symbol must be a product of factors, got {value!r}"
                    )
                powers[str(base)] = _normalize_exponent(exp)
            return cls(powers)
        raise TypeError(
            f"cannot build a UnitSymbol from {type(value)}; expected a str, "
            "dict, UnitSymbol, or a multiplicative sympy expression"
        )

    # -- explicitly reject addition / subtraction --------------------------
    # A UnitSymbol is multiplicative only; "m + s" has no unit meaning.
    def _no_addition(self, other: Any) -> "UnitSymbol":
        raise TypeError(
            "unit symbols cannot be added or subtracted : UnitSymbol is a "
            "multiplicative monomial (use *, / or ** only)"
        )

    __add__ = _no_addition
    __radd__ = _no_addition
    __sub__ = _no_addition
    __rsub__ = _no_addition

    def __mul__(self, other: Any) -> "UnitSymbol":
        other = UnitSymbol.coerce(other)
        powers = dict(self.powers)
        for key, value in other.powers.items():
            powers[key] = powers.get(key, 0) + value
        return UnitSymbol(powers)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "UnitSymbol":
        other = UnitSymbol.coerce(other)
        powers = dict(self.powers)
        for key, value in other.powers.items():
            powers[key] = powers.get(key, 0) - value
        return UnitSymbol(powers)

    def __rtruediv__(self, other: Any) -> "UnitSymbol":
        return UnitSymbol.coerce(other) / self

    def __pow__(self, exponent: Scalar) -> "UnitSymbol":
        return UnitSymbol(
            {key: value * exponent for key, value in self.powers.items()}
        )

    def __str__(self) -> str:
        # default "1" matches sympy's empty product (e.g. m/m -> 1)
        return format_power_dict(self.powers, "1")

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        try:
            other = UnitSymbol.coerce(other)
        except Exception:
            return NotImplemented
        return self.powers == other.powers

    def __hash__(self) -> int:
        return hash(frozenset(self.powers.items()))
