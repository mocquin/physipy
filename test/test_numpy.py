"""
Unit tests for the numpy __array_function__ handlers in
physipy.quantity._numpy.

These handlers are registered via @implements(...) so that the corresponding
numpy functions work on Quantity objects, but many were never actually
invoked by the test suite. The tests below call each one and assert on the
result value/dimension and on the dimension-checking branches.

A handful of handlers are currently broken; those are marked with
unittest.expectedFailure and documented inline so the bug is recorded without
breaking the suite (and so a future fix turns the test green / flags the
marker for removal).
"""

import unittest

import numpy as np

from physipy import Dimension, DimensionError, Quantity, m, s


def L(arr):
    """Helper: a length-dimensioned array Quantity."""
    return np.asarray(arr, dtype=float) * m


class TestFFT(unittest.TestCase):
    """The whole np.fft family drops, computes, and re-attaches the dimension."""

    FUNCS = [
        np.fft.fft2,
        np.fft.ifft2,
        np.fft.fftn,
        np.fft.ifftn,
        np.fft.rfft,
        np.fft.irfft,
        np.fft.rfft2,
        np.fft.irfft2,
        np.fft.rfftn,
        np.fft.irfftn,
        np.fft.hfft,
        np.fft.ihfft,
    ]

    def test_preserve_dimension(self):
        a = L([[1.0, 2.0], [3.0, 4.0]])
        for func in self.FUNCS:
            with self.subTest(func=func.__name__):
                res = func(a)
                self.assertIsInstance(res, Quantity)
                self.assertEqual(res.dimension, m.dimension)
                # value matches the raw numpy computation on the magnitudes
                np.testing.assert_allclose(res.value, func(a.value))


class TestPolyfitPolyval(unittest.TestCase):
    def test_polyfit_coefficient_dimensions(self):
        x = np.array([1.0, 2.0, 3.0]) * s
        y = np.array([2.0, 4.0, 6.0]) * m
        coefs = np.polyfit(x, y, 1)
        self.assertIsInstance(coefs, tuple)
        # slope has dimension m/s, intercept has dimension m
        self.assertEqual(coefs[0].dimension, (m / s).dimension)
        self.assertEqual(coefs[1].dimension, m.dimension)

    def test_polyval(self):
        x = np.array([1.0, 2.0, 3.0]) * s
        y = np.array([2.0, 4.0, 6.0]) * m
        coefs = np.polyfit(x, y, 1)
        val = np.polyval(coefs, 2 * s)
        self.assertIsInstance(val, Quantity)
        self.assertAlmostEqual(val.value, 4.0)
        self.assertEqual(val.dimension, m.dimension)


class TestCov(unittest.TestCase):
    def test_single_variable(self):
        res = np.cov(L([1.0, 2.0, 3.0]))
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, (m**2).dimension)

    def test_two_variables(self):
        res = np.cov(L([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]) * s)
        self.assertEqual(res.dimension, (m * s).dimension)


class TestSearchsorted(unittest.TestCase):
    def test_value(self):
        self.assertEqual(np.searchsorted(L([1.0, 2.0, 3.0]), 2.5 * m), 2)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.searchsorted(L([1.0, 2.0, 3.0]), 2.5 * s)


class TestMayShareMemory(unittest.TestCase):
    def test_branches(self):
        a = L([1.0, 2.0, 3.0])
        self.assertTrue(np.may_share_memory(a, a))
        self.assertFalse(np.may_share_memory(a, L([9.0])))
        # mixed Quantity / ndarray branches
        self.assertFalse(np.may_share_memory(a, np.array([9.0])))
        self.assertFalse(np.may_share_memory(np.array([9.0]), a))


class TestSinc(unittest.TestCase):
    def test_dimensionless(self):
        x = Quantity(np.array([0.0, 0.5]), Dimension(None))
        np.testing.assert_allclose(np.sinc(x), np.sinc(np.array([0.0, 0.5])))

    def test_dimensionful_raises(self):
        with self.assertRaises(DimensionError):
            np.sinc(L([1.0, 2.0]))


class TestPad(unittest.TestCase):
    def test_constant_values(self):
        res = np.pad(L([1.0, 2.0, 3.0]), 1, constant_values=0 * m)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [0.0, 1.0, 2.0, 3.0, 0.0])

    def test_constant_values_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.pad(L([1.0, 2.0, 3.0]), 1, constant_values=0 * s)


class TestCorrcoef(unittest.TestCase):
    def test_shape_and_type(self):
        res = np.corrcoef(L([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        # corrcoef is dimensionless -> plain ndarray
        self.assertEqual(res.shape, (2, 2))


class TestDiff(unittest.TestCase):
    def test_basic(self):
        res = np.diff(L([1.0, 3.0, 6.0]))
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [2.0, 3.0])


class TestIsclose(unittest.TestCase):
    def test_close(self):
        self.assertTrue(bool(np.isclose(1 * m, 1.0000001 * m)))

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.isclose(1 * m, 1 * s)

    def test_atol_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.isclose(1 * m, 1 * m, atol=1 * s)


class TestAllclose(unittest.TestCase):
    def test_close(self):
        a = L([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(a, a))

    def test_dimension_mismatch_raises(self):
        a = L([1.0, 2.0, 3.0])
        with self.assertRaises(DimensionError):
            np.allclose(a, a.value * s)


class TestInterp(unittest.TestCase):
    def test_value_and_dimension(self):
        xp = np.array([1.0, 2.0, 3.0]) * s
        fp = np.array([10.0, 20.0, 30.0]) * m
        res = np.interp(1.5 * s, xp, fp)
        self.assertEqual(res.dimension, m.dimension)
        self.assertAlmostEqual(res.value, 15.0)

    def test_x_dimension_mismatch_raises(self):
        xp = np.array([1.0, 2.0, 3.0]) * s
        fp = np.array([10.0, 20.0, 30.0]) * m
        with self.assertRaises(DimensionError):
            np.interp(1.5 * m, xp, fp)


class TestHistogram(unittest.TestCase):
    def test_no_range(self):
        hist, edges = np.histogram(L([1.0, 2.0, 3.0, 4.0]), bins=2)
        self.assertIsInstance(edges, Quantity)
        self.assertEqual(edges.dimension, m.dimension)

    def test_range_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.histogram(L([1.0, 2.0, 3.0]), bins=2, range=(0 * m, 5 * s))


class TestPercentile(unittest.TestCase):
    def test_value_and_dimension(self):
        res = np.percentile(L([1.0, 2.0, 3.0, 4.0]), 50)
        self.assertEqual(res.dimension, m.dimension)
        self.assertAlmostEqual(res.value, 2.5)


# ---------------------------------------------------------------------------
# Regression tests for three handlers that were previously broken on their
# (untested) dimension-checking / multi-return branches.
# ---------------------------------------------------------------------------


class TestLinalgLstsq(unittest.TestCase):
    """np.linalg.lstsq returns a 4-tuple, each element with its own dimension."""

    def test_returns_dimensioned_tuple(self):
        A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]) * m
        b = np.array([1.0, 2.0, 3.0]) * s
        solution, residuals, rank, singular_values = np.linalg.lstsq(
            A, b, rcond=None
        )
        self.assertEqual(solution.dimension, (s / m).dimension)
        self.assertEqual(residuals.dimension, (s**2).dimension)
        self.assertEqual(rank, 2)
        self.assertEqual(singular_values.dimension, m.dimension)
        # solution magnitudes match the raw numpy least-squares fit
        np.testing.assert_allclose(
            solution.value,
            np.linalg.lstsq(A.value, b.value, rcond=None)[0],
        )


class TestHistogramRange(unittest.TestCase):
    def test_unit_range(self):
        hist, edges = np.histogram(
            L([1.0, 2.0, 3.0]), bins=2, range=(0 * m, 4 * m)
        )
        self.assertIsInstance(edges, Quantity)
        self.assertEqual(edges.dimension, m.dimension)
        np.testing.assert_allclose(edges.value, [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(hist, [1, 2])

    def test_range_dimension_must_match_data(self):
        with self.assertRaises(DimensionError):
            np.histogram(L([1.0, 2.0, 3.0]), bins=2, range=(0 * s, 4 * s))


class TestDiffPrependAppend(unittest.TestCase):
    def test_prepend(self):
        res = np.diff(L([1.0, 3.0, 6.0]), prepend=0 * m)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [1.0, 2.0, 3.0])

    def test_append(self):
        res = np.diff(L([1.0, 3.0, 6.0]), append=10 * m)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [2.0, 3.0, 4.0])

    def test_prepend_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.diff(L([1.0, 2.0, 3.0]), prepend=0 * s)

    def test_append_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            np.diff(L([1.0, 2.0, 3.0]), append=0 * s)


class TestAxisReorder(unittest.TestCase):
    """np.moveaxis / swapaxes / rollaxis only permute axes: the values are
    reordered exactly like raw numpy, and dimension/symbol/favunit carry
    through unchanged."""

    def _arr(self):
        return L(np.arange(24).reshape(2, 3, 4))

    def test_moveaxis_value_and_dimension(self):
        a = self._arr()
        res = np.moveaxis(a, -1, 0)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.shape, (4, 2, 3))
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_array_equal(res.value, np.moveaxis(a.value, -1, 0))

    def test_moveaxis_multiple_axes(self):
        a = self._arr()
        res = np.moveaxis(a, [0, 1], [-1, -2])
        self.assertEqual(res.shape, (4, 3, 2))
        np.testing.assert_array_equal(
            res.value, np.moveaxis(a.value, [0, 1], [-1, -2])
        )

    def test_swapaxes(self):
        a = self._arr()
        res = np.swapaxes(a, 0, 2)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_array_equal(res.value, np.swapaxes(a.value, 0, 2))

    def test_rollaxis(self):
        a = self._arr()
        res = np.rollaxis(a, 2, 0)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_array_equal(res.value, np.rollaxis(a.value, 2, 0))

    def test_preserves_symbol_and_favunit(self):
        from physipy import units

        a = np.arange(24).reshape(2, 3, 4) * m
        a.symbol = "x"
        a.favunit = units["mm"]
        for func, args in [
            (np.moveaxis, (-1, 0)),
            (np.swapaxes, (0, 2)),
            (np.rollaxis, (2, 0)),
        ]:
            with self.subTest(func=func.__name__):
                res = func(a, *args)
                self.assertEqual(str(res.symbol), "x")
                self.assertEqual(str(res.favunit.symbol), "mm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
