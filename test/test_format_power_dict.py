"""
Tests for the pure-Python power-dict renderer that replaced sympy on the
``str``/``repr`` path of Dimension (physipy.quantity.dimension.format_power_dict).

The expected strings are hard-coded (an explicit oracle, independent of sympy)
so they keep pinning the format even once sympy becomes optional. An additional
brute-force parity check against sympy runs only when sympy is importable.
"""

import itertools
import unittest

from physipy import m, s
from physipy.quantity.dimension import format_power_dict


class TestFormatPowerDict(unittest.TestCase):
    def test_empty_returns_default(self):
        self.assertEqual(format_power_dict({}, "no-dimension"), "no-dimension")
        self.assertEqual(
            format_power_dict({"L": 0, "T": 0}, "DEF"), "DEF"
        )

    def test_single_positive(self):
        self.assertEqual(format_power_dict({"m": 1}, ""), "m")
        self.assertEqual(format_power_dict({"m": 2}, ""), "m**2")
        self.assertEqual(format_power_dict({"m": 3}, ""), "m**3")

    def test_single_negative_integer(self):
        # sympy quirk : -1 collapses to 1/x, but -2/-3 stay as x**(-n)
        self.assertEqual(format_power_dict({"T": -1}, ""), "1/T")
        self.assertEqual(format_power_dict({"L": -2}, ""), "L**(-2)")
        self.assertEqual(format_power_dict({"L": -3}, ""), "L**(-3)")

    def test_single_float(self):
        self.assertEqual(format_power_dict({"m": 0.5}, ""), "m**0.5")
        self.assertEqual(format_power_dict({"m": 1.0}, ""), "m**1.0")
        self.assertEqual(format_power_dict({"m": 1.5}, ""), "m**1.5")
        self.assertEqual(format_power_dict({"m": -0.5}, ""), "m**(-0.5)")
        self.assertEqual(format_power_dict({"m": -1.0}, ""), "m**(-1.0)")

    def test_numerator_only(self):
        self.assertEqual(format_power_dict({"I": 1, "L": 1, "M": 1}, ""), "I*L*M")
        self.assertEqual(
            format_power_dict({"L": 2, "M": 1}, ""), "L**2*M"
        )

    def test_numerator_and_denominator(self):
        self.assertEqual(format_power_dict({"L": 1, "T": -1}, ""), "L/T")
        self.assertEqual(format_power_dict({"L": 1, "T": -2}, ""), "L/T**2")
        self.assertEqual(
            format_power_dict({"M": 1, "L": 2, "T": -2}, ""), "L**2*M/T**2"
        )
        self.assertEqual(
            format_power_dict({"M": 1, "L": 2, "T": -3, "I": -1}, ""),
            "L**2*M/(I*T**3)",
        )

    def test_denominator_only(self):
        self.assertEqual(format_power_dict({"L": -1, "T": -1}, ""), "1/(L*T)")
        self.assertEqual(
            format_power_dict({"T": -2, "L": -3}, ""), "1/(L**3*T**2)"
        )

    def test_symbols_are_sorted(self):
        # insertion order must not matter : output is sorted by symbol name
        self.assertEqual(
            format_power_dict({"T": -2, "M": 1, "L": 2}, ""), "L**2*M/T**2"
        )

    def test_via_real_dimension_objects(self):
        self.assertEqual(str((m**2 / s).dimension), "L**2/T")
        self.assertEqual((m**2 / s).dimension.str_SI_unit(), "m**2/s")
        self.assertEqual(str((m / s).dimension), "L/T")
        self.assertEqual(repr(m * s), "<Quantity : 1 m*s, symbol=m*s>")


class TestSympyParityOracle(unittest.TestCase):
    """Brute-force parity against sympy, when available."""

    def test_matches_sympy_over_exponent_space(self):
        try:
            import sympy  # noqa: F401
        except ImportError:
            self.skipTest("sympy not available")

        syms = ["L", "M", "T", "I", "theta"]
        for n in range(0, 4):
            for combo in itertools.combinations(syms, n):
                for exps in itertools.product(range(-3, 4), repeat=n):
                    d = {k: e for k, e in zip(combo, exps)}
                    self.assertEqual(
                        format_power_dict(d, "X"), self._sympy_str(d, "X")
                    )

    @staticmethod
    def _sympy_str(power_dict, default):
        from sympy import Symbol

        output = 1
        for key, value in power_dict.items():
            output *= Symbol(key) ** value
        return default if output == 1 else str(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
