"""
Verify that the sympy-backed features of physipy.quantity.dimension are
*optional* :

 - B (compound-string parsing, e.g. Dimension("L**2/T")) is gated behind the
   "symbolic" extra and raises a clear ImportError when sympy is missing ;
 - C (LaTeX rendering) degrades gracefully : `_repr_latex_` returns None (so
   IPython falls back to the text repr) and `latex_SI_unit` raises a clear
   ImportError ;
 - the core (dict / single-symbol construction, dimensional algebra, and the
   pure-Python str/repr) keeps working with no sympy at all.

sympy is actually installed in the test environment, so its absence is
simulated by replacing the module-level ``require`` used on the B/C paths.
"""

import unittest
from unittest import mock

import physipy.quantity.dimension as dim
import physipy.quantity.utils as utils
from physipy import Dimension


def _no_sympy_require(package, extra):
    """Stand-in for _optional.require that behaves as if sympy was absent."""
    if package.split(".")[0] == "sympy":
        raise ImportError(
            f"'sympy' is required for this feature of physipy. Install it "
            f"with `pip install physipy[{extra}]` (or `pip install sympy`)."
        )
    import importlib

    return importlib.import_module(package)


class TestSympyOptional(unittest.TestCase):
    def setUp(self):
        # patch the `require` name in both modules that use it for B/C
        self._patchers = [
            mock.patch.object(dim, "require", _no_sympy_require),
            mock.patch.object(utils, "require", _no_sympy_require),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patchers])

    # --- core works with no sympy ---
    def test_dict_construction_works(self):
        self.assertEqual(Dimension({"L": 2}).dim_dict["L"], 2)

    def test_single_symbol_construction_works(self):
        self.assertEqual(Dimension("L").dim_dict["L"], 1)

    def test_str_and_repr_work(self):
        self.assertEqual(str(Dimension({"L": 2}) / Dimension("T")), "L**2/T")
        self.assertIn("Dimension", repr(Dimension({"L": 2})))

    def test_algebra_works(self):
        self.assertEqual(
            str((Dimension("L") * Dimension("L")) / Dimension("T")), "L**2/T"
        )

    # --- B is gated ---
    def test_compound_string_construction_is_gated(self):
        with self.assertRaises(ImportError):
            Dimension("L**2/T")

    def test_error_message_points_to_extra(self):
        with self.assertRaises(ImportError) as ctx:
            Dimension("L/T**2")
        self.assertIn("symbolic", str(ctx.exception))

    # --- C degrades / gates ---
    def test_repr_latex_returns_none(self):
        self.assertIsNone(Dimension({"L": 2})._repr_latex_())

    def test_latex_si_unit_is_gated(self):
        with self.assertRaises(ImportError):
            Dimension({"L": 2}).latex_SI_unit()


class TestSympyPresentStillWorks(unittest.TestCase):
    """Sanity : with the real `require`, the sympy features behave as before."""

    def test_compound_string(self):
        self.assertEqual(Dimension("L**2/T").dim_dict["L"], 2)
        self.assertEqual(Dimension("L**2/T").dim_dict["T"], -1)

    def test_repr_latex_is_a_latex_string(self):
        latex = Dimension({"L": 2})._repr_latex_()
        self.assertIsInstance(latex, str)
        self.assertTrue(latex.startswith("$") and latex.endswith("$"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
