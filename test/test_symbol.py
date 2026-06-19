"""
Tests for the pure-Python UnitSymbol that replaced sympy on the
Quantity.symbol path, and for the resulting symbol labels on real quantities.

UnitSymbol reproduces sympy's monomial behaviour (canonical printing,
``m*m`` -> ``m**2``, ``m/m`` -> ``1``) ; a brute-force parity check against
sympy runs only when sympy is importable.
"""

import itertools
import unittest

from physipy import Quantity, Dimension, m, s, units
from physipy.quantity._symbol import UnitSymbol


class TestUnitSymbol(unittest.TestCase):
    def test_atom(self):
        self.assertEqual(str(UnitSymbol("m")), "m")

    def test_multiply_same_atom_squares(self):
        self.assertEqual(str(UnitSymbol("m") * UnitSymbol("m")), "m**2")

    def test_divide_same_atom_cancels(self):
        self.assertEqual(str(UnitSymbol("m") / UnitSymbol("m")), "1")

    def test_product_of_atoms_is_sorted(self):
        self.assertEqual(str(UnitSymbol("N") * UnitSymbol("m")), "N*m")
        self.assertEqual(str(UnitSymbol("m") * UnitSymbol("N")), "N*m")

    def test_quotient(self):
        self.assertEqual(str(UnitSymbol("m") / UnitSymbol("s")), "m/s")

    def test_power(self):
        self.assertEqual(str(UnitSymbol("m") ** 2), "m**2")
        self.assertEqual(str(UnitSymbol("m") ** -1), "1/m")
        self.assertEqual(str(UnitSymbol("m") ** 0.5), "m**0.5")

    def test_equality_and_hash(self):
        self.assertEqual(UnitSymbol("m"), UnitSymbol("m"))
        self.assertNotEqual(UnitSymbol("m"), UnitSymbol("s"))
        self.assertEqual(
            hash(UnitSymbol("m") * UnitSymbol("s")),
            hash(UnitSymbol("s") * UnitSymbol("m")),
        )

    def test_coerce_from_str_and_dict(self):
        self.assertEqual(UnitSymbol.coerce("m"), UnitSymbol({"m": 1}))

    def test_coerce_rejects_unsupported_type(self):
        with self.assertRaises(TypeError):
            UnitSymbol.coerce(5)


class TestUnitSymbolRejectsAddition(unittest.TestCase):
    """UnitSymbol is multiplicative only : addition/subtraction must raise."""

    def test_add_raises(self):
        with self.assertRaises(TypeError):
            UnitSymbol("m") + UnitSymbol("s")

    def test_sub_raises(self):
        with self.assertRaises(TypeError):
            UnitSymbol("m") - UnitSymbol("s")

    def test_radd_raises(self):
        with self.assertRaises(TypeError):
            1 + UnitSymbol("m")

    def test_rsub_raises(self):
        with self.assertRaises(TypeError):
            1 - UnitSymbol("m")

    def test_coerce_rejects_additive_sympy(self):
        try:
            import sympy
        except ImportError:
            self.skipTest("sympy not available")
        with self.assertRaises(TypeError):
            UnitSymbol.coerce(sympy.Symbol("a") + sympy.Symbol("b"))

    def test_coerce_accepts_multiplicative_sympy(self):
        try:
            import sympy
        except ImportError:
            self.skipTest("sympy not available")
        sym = UnitSymbol.coerce(sympy.Symbol("m") ** 2 / sympy.Symbol("s"))
        self.assertEqual(str(sym), "m**2/s")

    def test_quantity_symbol_setter_rejects_additive_sympy(self):
        try:
            import sympy
        except ImportError:
            self.skipTest("sympy not available")
        q = 1 * m
        with self.assertRaises(TypeError):
            q.symbol = sympy.Symbol("a") + sympy.Symbol("b")


class TestQuantitySymbolLabels(unittest.TestCase):
    """The user-visible symbol labels must be unchanged."""

    def test_known_labels(self):
        N = units["N"]
        self.assertEqual(str((m * m).symbol), "m**2")
        self.assertEqual(str((m * s).symbol), "m*s")
        self.assertEqual(str((m / s).symbol), "m/s")
        self.assertEqual(str((N * m).symbol), "N*m")
        self.assertEqual(str((m**2).symbol), "m**2")
        self.assertEqual(str((m**-1).symbol), "1/m")
        self.assertEqual(str(((N * m) / s).symbol), "N*m/s")
        self.assertEqual(str((2 * m).symbol), "UndefinedSymbol*m")

    def test_default_symbol(self):
        self.assertEqual(str(Quantity(1, Dimension("L")).symbol), "UndefinedSymbol")

    def test_repr_includes_symbol(self):
        self.assertEqual(repr(m * m), "<Quantity : 1 m**2, symbol=m**2>")

    def test_set_symbol(self):
        q = (10 * s).set_symbol("period")
        self.assertEqual(str(q.symbol), "period")


class TestSympyParityOracle(unittest.TestCase):
    def test_matches_sympy(self):
        try:
            from sympy import Symbol
        except ImportError:
            self.skipTest("sympy not available")

        atoms = ["m", "s", "N", "kg", "UndefinedSymbol"]
        for n in range(0, 4):
            for combo in itertools.combinations(atoms, n):
                for exps in itertools.product(range(-3, 4), repeat=n):
                    d = {k: e for k, e in zip(combo, exps) if e != 0}
                    out = 1
                    for k, e in d.items():
                        out *= Symbol(k) ** e
                    self.assertEqual(str(UnitSymbol(d)), str(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
