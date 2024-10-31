import time
import unittest

import numpy as np

from physipy import m


class TestNumpyFunctions(unittest.TestCase):
    def setUp(self):
        self.arr_1d = np.array([1, 2, np.nan, 4, 5])
        self.arr_2d = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
        self.arr_3d = np.array(
            [[[1, 2, np.nan], [3, 4, 5]], [[6, 7, 8], [9, 10, np.nan]]]
        )
        self.arr_3d_extra = m * np.array(
            [
                [[1, 2, np.nan, 4], [5, 6, 7, 8], [9, 10, 11, np.nan]],
                [[13, 14, 15, 16], [17, np.nan, 19, 20], [21, 22, 23, np.nan]],
            ]
        )

        self.arr_singleton = np.array([np.nan])

        self.startTime = time.time()
        self.tottime = 0

    def tearDown(self):
        t = time.time() - self.startTime
        self.tottime = self.tottime + t
        # print(f"{self.id():70} : {t:10.6f}")
        self.times.append(t)
        self.ids.append(str(self.id()))

    @classmethod
    def setUpClass(self):
        self.arr = np.array([1, 2, np.nan, 4, 5]) * m
        self.times = []
        self.ids = []

    def test_nanmin(self):
        expected_min = 1 * m
        self.assertAlmostEqual(np.nanmin(self.arr), expected_min)

    def test_nanmax(self):
        expected_max = 5 * m
        self.assertAlmostEqual(np.nanmax(self.arr), expected_max)

    def test_nanargmin(self):
        expected_argmin = 0
        self.assertEqual(np.nanargmin(self.arr), expected_argmin)

    def test_nanargmax(self):
        expected_argmax = 4
        self.assertEqual(np.nanargmax(self.arr), expected_argmax)

    def test_nansum(self):
        expected_sum = 12 * m
        self.assertAlmostEqual(np.nansum(self.arr), expected_sum)

    def test_nanmean(self):
        expected_mean = 3 * m
        self.assertAlmostEqual(np.nanmean(self.arr), expected_mean)

    def test_nanmedian(self):
        expected_median = 3 * m
        self.assertAlmostEqual(np.nanmedian(self.arr), expected_median)

    def test_nanvar(self):
        expected_var = 2.5 * m**2
        self.assertAlmostEqual(np.nanvar(self.arr), expected_var)

    def test_nanstd(self):
        expected_std = 1.5811388300841898 * m
        self.assertAlmostEqual(np.nanstd(self.arr), expected_std)

    def test_nanpercentile(self):
        expected_percentile = 3 * m
        self.assertAlmostEqual(
            np.nanpercentile(self.arr, 50), expected_percentile
        )

    def test_nanprod(self):
        # 4 or 5 ? should nan's dimension be ignored ?
        expected_prod = 40 * m**5
        self.assertAlmostEqual(np.nanprod(self.arr), expected_prod)

    def test_nancumprod(self):
        with self.assertRaises(TypeError):
            np.nancumprod(self.arr)

    def test_nancumsum(self):
        expected_cumsum = np.array([1, 3, 3, 7, 12]) * m
        self.assertTrue(
            np.allclose(
                np.nancumsum(self.arr), expected_cumsum, atol=1e-08 * m
            )
        )

    def test_nanmin_1d(self):
        expected_min = 1
        self.assertAlmostEqual(np.nanmin(self.arr_1d), expected_min)

    def test_nanmin_2d_axis0(self):
        expected_min = [1, 2, 6]
        res = np.nanmin(self.arr_2d, axis=0)
        self.assertTrue(np.allclose(res, expected_min))

    def test_nanmin_2d_axis1(self):
        expected_min = [1, 4, 7]
        res = np.nanmin(self.arr_2d, axis=1)
        self.assertTrue(np.allclose(res, expected_min))

    def test_nanmin_3d_axis0(self):
        expected_min = [[1, 2, 8], [3, 4, 5]]
        res = np.nanmin(self.arr_3d, axis=0)
        self.assertTrue(np.allclose(res, expected_min))

    def test_nanmin_singleton(self):
        result = np.nanmin(self.arr_singleton)
        self.assertTrue(np.isnan(result))

    def test_nanmin_3d_extra_axis0(self):
        expected_min = [[1, 2, 15, 4], [5, 6, 7, 8], [9, 10, 11, np.nan]] * m
        res = np.nanmin(self.arr_3d_extra, axis=0)
        self.assertTrue(np.array_equal(res, expected_min, equal_nan=True))

    def test_nanmin_3d_extra_axis_1_2(self):
        expected_min = [1, 13] * m
        res = np.nanmin(self.arr_3d_extra, axis=(1, 2))
        self.assertTrue(np.allclose(res, expected_min, atol=1e-08 * m))

    def test_nanmin_3d_extra_axis_none(self):
        expected_min = 1 * m
        self.assertAlmostEqual(
            np.nanmin(self.arr_3d_extra, axis=None), expected_min
        )

    def test_nanvar_3d_extra_axis_1_2(self):
        res = np.nanvar(self.arr_3d_extra, axis=(1, 2))
        exp = np.nanvar(self.arr_3d_extra.value, axis=(1, 2)) * m**2
        self.assertTrue(np.allclose(res, exp, atol=1e-08 * m**2))

    def test_nanprod_3d_extra_axis_1_2(self):
        res = np.nanprod(self.arr_3d_extra, axis=(1, 2))
        exp = np.nanprod(self.arr_3d_extra.value, axis=(1, 2)) * m**12
        self.assertTrue(np.allclose(res, exp, atol=1e-08 * m**12))

    def test_nanprod_3d_extra_axis_0(self):
        res = np.nanprod(self.arr_3d_extra, axis=0)
        exp = np.nanprod(self.arr_3d_extra.value, axis=0) * m**2
        self.assertTrue(np.allclose(res, exp, atol=1e-08 * m**2))

    def test_nanprod_3d_extra_axis_None(self):
        res = np.nanprod(self.arr_3d_extra)
        exp = np.nanprod(self.arr_3d_extra.value) * m**24
        self.assertTrue(np.allclose(res, exp, atol=1e-08 * m**24))


if __name__ == "__main__":
    unittest.main()
