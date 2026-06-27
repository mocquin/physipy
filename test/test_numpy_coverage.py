"""
Tests for the public numpy-coverage introspection API in
physipy.quantity._numpy : ``supported_numpy_functions`` and ``numpy_coverage``.

These don't pin exact counts (those drift with the running numpy version and as
handlers get added); they assert the *invariants* of the report -- partitions
are complete and disjoint, known functions land in the right bucket, and the
two public entry points agree with each other.
"""

import unittest

import numpy as np

import physipy
from physipy import numpy_coverage, supported_numpy_functions
from physipy.quantity._numpy import (
    HANDLED_FUNCTIONS,
    NumpyCoverage,
    _canonical_numpy_ufuncs,
    _public_numpy_functions,
    implemented_ufuncs,
)


class TestSupportedNumpyFunctions(unittest.TestCase):
    def test_exposed_at_top_level(self):
        # re-exported from both physipy and physipy.quantity
        self.assertIs(physipy.supported_numpy_functions, supported_numpy_functions)
        self.assertIs(physipy.numpy_coverage, numpy_coverage)
        self.assertIs(physipy.NumpyCoverage, NumpyCoverage)

    def test_returns_set_of_callables(self):
        sup = supported_numpy_functions()
        self.assertIsInstance(sup, set)
        self.assertTrue(sup)
        self.assertTrue(all(callable(f) for f in sup))

    def test_contains_known_array_functions_and_ufuncs(self):
        sup = supported_numpy_functions()
        for f in (np.concatenate, np.unique, np.linalg.norm, np.fft.fft):
            self.assertIn(f, sup)
        for uf in (np.add, np.sin, np.sqrt):
            self.assertIn(uf, sup)

    def test_excludes_unimplemented(self):
        sup = supported_numpy_functions()
        self.assertNotIn(np.einsum, sup)
        self.assertNotIn(np.bitwise_and, sup)

    def test_names_variant_is_sorted_strings(self):
        names = supported_numpy_functions(names=True)
        self.assertIsInstance(names, list)
        self.assertTrue(all(isinstance(n, str) for n in names))
        self.assertEqual(names, sorted(names))
        self.assertIn("add", names)
        self.assertIn("concatenate", names)

    def test_unifies_both_dispatch_mechanisms(self):
        sup = supported_numpy_functions()
        # every __array_function__ handler is supported ...
        self.assertTrue(set(HANDLED_FUNCTIONS) <= sup)
        # ... and so is every implemented ufunc that exists in this numpy.
        for name in implemented_ufuncs:
            if hasattr(np, name):
                self.assertIn(getattr(np, name), sup)


class TestNumpyCoverage(unittest.TestCase):
    def setUp(self):
        self.cov = numpy_coverage()

    def test_type_and_version(self):
        self.assertIsInstance(self.cov, NumpyCoverage)
        self.assertEqual(self.cov.numpy_version, np.__version__)

    def test_ufunc_partition_is_complete_and_disjoint(self):
        grp = self.cov.ufuncs
        impl, miss, na = set(grp.implemented), set(grp.missing), set(grp.not_applicable)
        # pairwise disjoint
        self.assertFalse(impl & miss)
        self.assertFalse(impl & na)
        self.assertFalse(miss & na)
        # together they cover exactly the canonical (alias-deduped) ufunc set
        self.assertEqual(impl | miss | na, set(_canonical_numpy_ufuncs()))

    def test_array_function_partition_is_complete_and_disjoint(self):
        grp = self.cov.array_functions
        impl, miss, na = (
            set(grp.implemented),
            set(grp.missing),
            set(grp.not_applicable),
        )
        self.assertFalse(impl & miss)
        self.assertFalse(impl & na)
        self.assertFalse(miss & na)
        self.assertEqual(impl | miss | na, set(_public_numpy_functions()))

    def test_known_buckets(self):
        uf, fn = self.cov.ufuncs, self.cov.array_functions
        self.assertIn("add", uf.implemented)
        self.assertIn("sin", uf.implemented)
        self.assertIn("concatenate", fn.implemented)
        self.assertIn("linalg.norm", fn.implemented)
        # not yet implemented
        self.assertIn("einsum", fn.missing)
        # declared impossible -> not_applicable, never missing/implemented
        self.assertIn("logical_and", uf.not_applicable)
        self.assertNotIn("logical_and", uf.missing)
        # heterogeneous-output array funcs can't fit the single-Dimension model
        self.assertIn("vander", fn.not_applicable)
        self.assertNotIn("vander", fn.missing)

    def test_alias_dedup(self):
        # numpy>=2.0 aliases collapse onto their canonical name and are not
        # double-counted (np.abs is np.absolute, np.mod is np.remainder).
        uf = self.cov.ufuncs
        allnames = set(uf.implemented) | set(uf.missing) | set(uf.not_applicable)
        self.assertNotIn("abs", allnames)
        self.assertIn("absolute", uf.implemented)
        self.assertNotIn("mod", allnames)
        self.assertIn("remainder", uf.implemented)

    def test_ratio_bounds_and_formula(self):
        for grp in (self.cov.ufuncs, self.cov.array_functions):
            self.assertEqual(grp.n_relevant, len(grp.implemented) + len(grp.missing))
            self.assertTrue(0.0 <= grp.ratio <= 1.0)
            if grp.n_relevant:
                self.assertAlmostEqual(
                    grp.ratio, len(grp.implemented) / grp.n_relevant
                )

    def test_implemented_names_are_actually_supported(self):
        # cross-check the report against supported_numpy_functions
        sup = supported_numpy_functions()
        public = _public_numpy_functions()
        for name in self.cov.array_functions.implemented:
            self.assertIn(public[name], sup)

    def test_summary_is_str(self):
        s = self.cov.summary()
        self.assertIsInstance(s, str)
        self.assertIn("numpy coverage", s)
        self.assertEqual(str(self.cov), s)

    def test_to_markdown(self):
        md = self.cov.to_markdown()
        self.assertIsInstance(md, str)
        # both family sections are present
        self.assertIn("### Array functions", md)
        self.assertIn("### Ufuncs", md)
        # the three buckets are rendered with their counts
        fn = self.cov.array_functions
        self.assertIn(f"**Implemented ({len(fn.implemented)}):**", md)
        self.assertIn(f"**Missing ({len(fn.missing)}):**", md)
        self.assertIn(f"**Not applicable ({len(fn.not_applicable)}):**", md)
        # known names land in the rendered text
        self.assertIn("`concatenate`", md)
        self.assertIn("`vander`", md)
        self.assertIn(self.cov.numpy_version, md)


if __name__ == "__main__":
    unittest.main()
