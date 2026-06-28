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


class TestPtp(unittest.TestCase):
    """np.ptp (peak-to-peak) keeps the input's dimension."""

    def test_value_and_dimension(self):
        a = L([1.0, 5.0, 2.0, 8.0, 3.0])
        res = np.ptp(a)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        self.assertAlmostEqual(res.value, 7.0)

    def test_axis(self):
        res = np.ptp(L([[1.0, 5.0], [2.0, 3.0]]), axis=1)
        np.testing.assert_allclose(res.value, [4.0, 1.0])
        self.assertEqual(res.dimension, m.dimension)


class TestQuantile(unittest.TestCase):
    """np.quantile / np.nanquantile keep the input's dimension (q is a fraction)."""

    def test_quantile(self):
        res = np.quantile(L([1.0, 2.0, 3.0, 4.0]), 0.5)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        self.assertAlmostEqual(res.value, 2.5)

    def test_nanquantile(self):
        res = np.nanquantile(L([1.0, np.nan, 3.0]), 0.5)
        self.assertEqual(res.dimension, m.dimension)
        self.assertAlmostEqual(res.value, 2.0)

    def test_matches_percentile(self):
        a = L([3.0, 1.0, 4.0, 1.0, 5.0])
        self.assertAlmostEqual(
            np.quantile(a, 0.25).value, np.percentile(a, 25).value
        )


class TestImag(unittest.TestCase):
    """np.imag mirrors np.real : keeps the dimension, takes the imaginary part."""

    def test_complex(self):
        c = np.array([1 + 2j, 3 - 1j]) * m
        res = np.imag(c)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [2.0, -1.0])

    def test_real_input_gives_zeros(self):
        res = np.imag(L([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(res.value, [0.0, 0.0, 0.0])
        self.assertEqual(res.dimension, m.dimension)


class TestCumulative(unittest.TestCase):
    """Cumulative reductions : sum keeps the dimension, prod requires dimensionless."""

    def test_cumulative_sum_keeps_dimension(self):
        a = L([1.0, 2.0, 3.0, 4.0])
        for func in (np.cumsum, np.cumulative_sum):
            with self.subTest(func=func.__name__):
                res = func(a)
                self.assertIsInstance(res, Quantity)
                self.assertEqual(res.dimension, m.dimension)
                np.testing.assert_allclose(res.value, [1.0, 3.0, 6.0, 10.0])

    def test_cumprod_dimensionless_ok(self):
        d = Quantity(np.array([1.0, 2.0, 3.0, 4.0]), Dimension(None))
        for func in (np.cumprod, np.cumulative_prod):
            with self.subTest(func=func.__name__):
                res = func(d)
                self.assertIsInstance(res, Quantity)
                self.assertTrue(res.is_dimensionless())
                np.testing.assert_allclose(res.value, [1.0, 2.0, 6.0, 24.0])

    def test_cumprod_dimensionful_raises(self):
        a = L([1.0, 2.0, 3.0])
        for func in (np.cumprod, np.cumulative_prod):
            with self.subTest(func=func.__name__):
                with self.assertRaises(DimensionError):
                    func(a)


class TestSplit(unittest.TestCase):
    """The split family returns a list of sub-Quantities, each keeping the
    input's dimension (and favunit)."""

    def test_split_and_array_split(self):
        a = L([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        for func, arg, n in [(np.split, 3, 3), (np.array_split, 4, 4)]:
            with self.subTest(func=func.__name__):
                parts = func(a, arg)
                self.assertIsInstance(parts, list)
                self.assertEqual(len(parts), n)
                for p, raw in zip(parts, func(a.value, arg)):
                    self.assertIsInstance(p, Quantity)
                    self.assertEqual(p.dimension, m.dimension)
                    np.testing.assert_allclose(p.value, raw)

    def test_hsplit_vsplit_dsplit(self):
        m2 = (np.arange(16.0).reshape(4, 4)) * m
        for func in (np.hsplit, np.vsplit):
            with self.subTest(func=func.__name__):
                parts = func(m2, 2)
                self.assertEqual(len(parts), 2)
                self.assertTrue(all(p.dimension == m.dimension for p in parts))
        m3 = (np.arange(8.0).reshape(2, 2, 2)) * m
        parts = np.dsplit(m3, 2)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(p.dimension == m.dimension for p in parts))

    def test_preserves_favunit(self):
        from physipy import units

        a = np.arange(6.0) * m
        a.favunit = units["mm"]
        for part in np.split(a, 3):
            self.assertEqual(str(part.favunit.symbol), "mm")


class TestDelete(unittest.TestCase):
    """np.delete drops elements while keeping the dimension (and favunit)."""

    def test_1d(self):
        a = L([0.0, 1.0, 2.0, 3.0, 4.0])
        res = np.delete(a, [1, 3])
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [0.0, 2.0, 4.0])

    def test_axis(self):
        res = np.delete(np.arange(9.0).reshape(3, 3) * m, 1, axis=0)
        self.assertEqual(res.shape, (2, 3))
        self.assertEqual(res.dimension, m.dimension)

    def test_preserves_favunit(self):
        from physipy import units

        a = np.arange(4.0) * m
        a.favunit = units["mm"]
        self.assertEqual(str(np.delete(a, 0).favunit.symbol), "mm")


class TestTriangularDiagflat(unittest.TestCase):
    """tril / triu / diagflat keep the input's dimension."""

    def test_tril_triu(self):
        M = np.arange(9.0).reshape(3, 3) * m
        for func in (np.tril, np.triu):
            for k in (-1, 0, 1):
                with self.subTest(func=func.__name__, k=k):
                    res = func(M, k)
                    self.assertIsInstance(res, Quantity)
                    self.assertEqual(res.dimension, m.dimension)
                    np.testing.assert_allclose(res.value, func(M.value, k))

    def test_diagflat(self):
        res = np.diagflat(L([1.0, 2.0, 3.0]))
        self.assertEqual(res.shape, (3, 3))
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(np.diag(res.value), [1.0, 2.0, 3.0])


class TestDimensionPreservingTransforms(unittest.TestCase):
    """Tier-1 transforms : run on the magnitudes and keep the input dimension."""

    def test_value_preserving_funcs(self):
        a = L([1.0, 3.0, 6.0, 10.0])
        cases = [
            (np.ediff1d, (), [2.0, 3.0, 4.0]),
            (np.fix, (), np.fix(a.value)),
            (np.trim_zeros, (), a.value),
            (np.nan_to_num, (), a.value),
            (np.sort_complex, (), np.sort_complex(a.value)),
            (np.real_if_close, (), a.value),
        ]
        for func, args, expected in cases:
            with self.subTest(func=func.__name__):
                res = func(a, *args)
                self.assertIsInstance(res, Quantity)
                self.assertEqual(res.dimension, m.dimension)
                np.testing.assert_allclose(res.value, expected)

    def test_ediff1d_to_end_begin(self):
        res = np.ediff1d(L([1.0, 3.0]), to_end=5.0 * m, to_begin=0.0 * m)
        np.testing.assert_allclose(res.value, [0.0, 2.0, 5.0])
        with self.assertRaises(DimensionError):
            np.ediff1d(L([1.0, 2.0]), to_end=5.0 * s)

    def test_resize_take_along_axis(self):
        self.assertEqual(np.resize(L([1.0, 2.0, 3.0]), (2, 3)).shape, (2, 3))
        res = np.take_along_axis(L([10.0, 20.0, 30.0]), np.array([2, 0]), axis=0)
        np.testing.assert_allclose(res.value, [30.0, 10.0])
        self.assertEqual(res.dimension, m.dimension)

    def test_unstack(self):
        parts = np.unstack(L([[1.0, 2.0], [3.0, 4.0]]), axis=0)
        self.assertIsInstance(parts, tuple)
        self.assertTrue(all(p.dimension == m.dimension for p in parts))
        np.testing.assert_allclose(parts[1].value, [3.0, 4.0])

    def test_block(self):
        res = np.block([L([1.0, 2.0]), L([3.0, 4.0])])
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(DimensionError):
            np.block([L([1.0]), np.array([2.0]) * s])

    def test_extract_choose(self):
        res = np.extract(np.array([True, False, True]), L([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(res.value, [1.0, 3.0])
        res = np.choose(np.array([0, 1, 0]), [L([1.0, 2.0, 3.0]), L([4.0, 5.0, 6.0])])
        np.testing.assert_allclose(res.value, [1.0, 5.0, 3.0])
        self.assertEqual(res.dimension, m.dimension)
        with self.assertRaises(DimensionError):
            np.choose(np.array([0, 1]), [L([1.0, 2.0]), np.array([3.0, 4.0]) * s])


class TestIndexReturning(unittest.TestCase):
    """nonzero / argwhere / flatnonzero drop the unit (indices are plain ints)."""

    def test_indices(self):
        a = L([0.0, 1.0, 0.0, 2.0])
        nz = np.nonzero(a)
        self.assertIsInstance(nz, tuple)
        np.testing.assert_array_equal(nz[0], [1, 3])
        np.testing.assert_array_equal(np.flatnonzero(a), [1, 3])
        np.testing.assert_array_equal(np.argwhere(a).ravel(), [1, 3])


class TestInPlaceWrites(unittest.TestCase):
    """put / putmask / place / put_along_axis mutate through `.value`, return None."""

    def test_put(self):
        a = L([1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(np.put(a, [0, 2], 9.0 * m))
        np.testing.assert_allclose(a.value, [9.0, 2.0, 9.0, 4.0])

    def test_putmask(self):
        a = L([1.0, 2.0, 3.0, 4.0])
        np.putmask(a, np.array([True, False, True, False]), 0.0 * m)
        np.testing.assert_allclose(a.value, [0.0, 2.0, 0.0, 4.0])

    def test_place(self):
        a = L([1.0, 2.0, 3.0, 4.0])
        np.place(a, a > 2.0 * m, np.array([7.0, 8.0]) * m)
        np.testing.assert_allclose(a.value, [1.0, 2.0, 7.0, 8.0])

    def test_put_along_axis(self):
        a = L([[1.0, 2.0], [3.0, 4.0]])
        np.put_along_axis(a, np.array([[0], [1]]), 0.0 * m, axis=1)
        np.testing.assert_allclose(a.value, [[0.0, 2.0], [3.0, 0.0]])

    def test_dimension_mismatch_raises(self):
        for func in (
            lambda: np.put(L([1.0, 2.0]), [0], 9.0 * s),
            lambda: np.putmask(L([1.0, 2.0]), np.array([True, False]), 9.0 * s),
            lambda: np.place(L([1.0, 2.0]), np.array([True, False]), 9.0 * s),
        ):
            with self.assertRaises(DimensionError):
                func()


class TestFullLikeQuantityFill(unittest.TestCase):
    """Gotcha (issue #26): numpy dispatches `full_like` on its template and
    `full` on `like=`, never on `fill_value` -- so a Quantity fill with a plain
    template isn't honoured. These pin the documented reliable workarounds."""

    def test_template_carrying_unit_works(self):
        res = np.full_like(np.arange(3) * m, 3 * m)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [3.0, 3.0, 3.0])

    def test_multiply_unitless_template_works(self):
        res = np.ones_like(np.arange(3)) * (3 * m)
        self.assertIsInstance(res, Quantity)
        self.assertEqual(res.dimension, m.dimension)
        np.testing.assert_allclose(res.value, [3.0, 3.0, 3.0])


class TestUfuncOut(unittest.TestCase):
    """``out=`` handling for the ufunc (``__array_ufunc__``) dispatch.

    Before the fix ``out=`` was silently dropped on the ``__call__`` path
    (buffer never written, ``r is out`` False) and crashed the ``reduce`` /
    ``accumulate`` paths with an ``AttributeError``.
    """

    def test_binary_out_quantity_matching_dim(self):
        a = L([1.0, 2.0, 3.0])
        out = np.zeros(3) * m
        buf_id = id(out.value)
        r = np.add(a, a, out=out)
        # numpy contract: the return value *is* the out target
        self.assertIs(r, out)
        # written in place (same underlying buffer), correct values
        self.assertEqual(id(out.value), buf_id)
        np.testing.assert_allclose(out.value, [2.0, 4.0, 6.0])
        self.assertEqual(out.dimension, m.dimension)

    def test_multiply_out_quantity_result_dim(self):
        a = L([1.0, 2.0, 3.0])
        out = np.zeros(3) * m**2
        r = np.multiply(a, a, out=out)
        self.assertIs(r, out)
        np.testing.assert_allclose(out.value, [1.0, 4.0, 9.0])
        self.assertEqual(out.dimension, (m**2).dimension)

    def test_unary_sqrt_out_quantity(self):
        a = L([1.0, 4.0, 9.0])
        out = np.zeros(3) * m**0.5
        r = np.sqrt(a, out=out)
        self.assertIs(r, out)
        np.testing.assert_allclose(out.value, [1.0, 2.0, 3.0])
        self.assertEqual(out.dimension, (m**0.5).dimension)

    def test_out_quantity_wrong_dim_raises(self):
        a = L([1.0, 2.0, 3.0])
        with self.assertRaises(DimensionError):
            np.add(a, a, out=np.zeros(3) * s)

    def test_out_plain_ndarray_dimensioned_raises(self):
        # a bare ndarray cannot carry a Dimension -> refuse (astropy stance)
        a = L([1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            np.add(a, a, out=np.zeros(3))

    def test_reduce_out_quantity(self):
        # the reduce path used to crash with AttributeError on out=Quantity
        a = L([1.0, 2.0, 3.0])
        out = np.zeros(()) * m
        r = np.add.reduce(a, out=out)
        self.assertIs(r, out)
        np.testing.assert_allclose(np.asarray(out.value), 6.0)
        self.assertEqual(out.dimension, m.dimension)

    def test_accumulate_out_quantity(self):
        a = L([1.0, 2.0, 3.0])
        out = np.zeros(3) * m
        r = np.add.accumulate(a, out=out)
        self.assertIs(r, out)
        np.testing.assert_allclose(out.value, [1.0, 3.0, 6.0])
        self.assertEqual(out.dimension, m.dimension)

    def test_comparison_out_plain_bool_buffer(self):
        # dimensionless result -> a plain ndarray out is fine and aliased
        a = L([1.0, 2.0, 3.0])
        out = np.zeros(3, dtype=bool)
        r = np.greater(a, 2 * m, out=out)
        self.assertIs(r, out)
        np.testing.assert_array_equal(out, [False, False, True])

    def test_no_out_is_unchanged(self):
        # regression: omitting out= still returns a fresh Quantity
        a = L([1.0, 2.0, 3.0])
        r = np.add(a, a)
        self.assertIsInstance(r, Quantity)
        self.assertIsNot(r, a)
        np.testing.assert_allclose(r.value, [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
