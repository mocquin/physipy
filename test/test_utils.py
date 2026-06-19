"""
Unit tests for helpers in physipy.quantity.utils.

Focus is on the equality helpers (hard_equal / very_hard_equal) and asqarray,
which previously had bugs on their array / asymmetric-favunit / mixed-dimension
branches.
"""

import unittest

import numpy as np

from physipy import DimensionError, Quantity, m, s
from physipy.quantity.utils import asqarray, hard_equal, very_hard_equal


class TestHardEqual(unittest.TestCase):
    def test_scalars(self):
        self.assertTrue(hard_equal(1 * m, 1 * m))
        self.assertFalse(hard_equal(1 * m, 2 * m))

    def test_dimension_mismatch(self):
        self.assertFalse(hard_equal(1 * m, 1 * s))

    def test_equal_arrays(self):
        a = np.array([1.0, 2.0, 3.0]) * m
        b = np.array([1.0, 2.0, 3.0]) * m
        self.assertTrue(hard_equal(a, b))

    def test_unequal_arrays(self):
        a = np.array([1.0, 2.0, 3.0]) * m
        c = np.array([1.0, 9.0, 3.0]) * m
        self.assertFalse(hard_equal(a, c))


class TestVeryHardEqual(unittest.TestCase):
    def _mm(self):
        unit = 1 * m
        unit.symbol = "mm"
        return unit

    def test_both_no_favunit(self):
        self.assertTrue(very_hard_equal(2 * m, 2 * m))

    def test_both_same_favunit(self):
        x = 2 * m
        x.favunit = self._mm()
        y = 2 * m
        y.favunit = self._mm()
        self.assertTrue(very_hard_equal(x, y))

    def test_asymmetric_favunit_returns_false(self):
        # one has a favunit, the other does not : must be False, not crash.
        x = 2 * m
        y = 2 * m
        y.favunit = self._mm()
        self.assertFalse(very_hard_equal(x, y))
        self.assertFalse(very_hard_equal(y, x))


class TestAsqarray(unittest.TestCase):
    def test_list_of_quantities(self):
        res = asqarray([1 * m, 2 * m, 3 * m])
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_array_equal(res.value, [1, 2, 3])

    def test_nested_list(self):
        res = asqarray([[1 * m, 2 * m], [3 * m, 4 * m]])
        self.assertEqual(res.value.shape, (2, 2))
        self.assertEqual(res.dimension, m.dimension)

    def test_list_mixed_dimensions_raises(self):
        with self.assertRaises(DimensionError):
            asqarray([1 * m, 2 * s])

    def test_ndarray_mixed_dimensions_raises(self):
        # this branch previously raised AttributeError (q.dim typo) instead
        # of a clean DimensionError.
        arr = np.array([1.0 * m, 2.0 * s], dtype=object)
        with self.assertRaises(DimensionError):
            asqarray(arr)

    def test_ndarray_of_quantities(self):
        arr = np.array([1.0 * m, 2.0 * m, 3.0 * m], dtype=object)
        res = asqarray(arr)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_array_equal(res.value, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
