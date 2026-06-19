"""
Unit tests for physipy.calculus, the Quantity-aware wrappers around
scipy.integrate / scipy.optimize.

Focus is on the integration/ODE/root helpers that were previously only
exercised (if at all) through doctests: result value & dimension, and the
dimension-checking branches.
"""

import unittest

import numpy as np

from physipy import DimensionError, Quantity, m, s
from physipy.calculus import (
    brentq,
    dblquad,
    quad,
    quad_vec,
    root,
    solve_ivp,
    tplquad,
    trapz2,
)


class TestTrapz2(unittest.TestCase):
    def test_uniform_area(self):
        # Integrate a constant 1 over a 2m x 1m domain -> 2 m**2.
        ech_x = np.linspace(0 * m, 2 * m, num=12)
        ech_y = np.linspace(0 * m, 1 * m, num=30)
        X, _ = np.meshgrid(ech_x, ech_y)
        Zs = np.ones_like(X)
        res = trapz2(Zs, ech_x, ech_y)
        self.assertIsInstance(res, Quantity)
        self.assertAlmostEqual(res.value, 2.0)
        self.assertEqual(res.dimension, (m**2).dimension)


class TestQuad(unittest.TestCase):
    def test_value_and_dimension(self):
        # integral of x dx from 0 to 2 m = 2 m**2
        res, _ = quad(lambda x: x, 0 * m, 2 * m)
        self.assertAlmostEqual(res.value, 2.0)
        self.assertEqual(res.dimension, (m**2).dimension)

    def test_dimensionless_result_is_stripped(self):
        # integrand dimensionless, bound dimensionless -> plain number
        res, _ = quad(lambda x: x, 0.0, 2.0)
        self.assertNotIsInstance(res, Quantity)
        self.assertAlmostEqual(res, 2.0)

    def test_bound_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            quad(lambda x: x, 0 * m, 1 * s)


class TestQuadVec(unittest.TestCase):
    def test_vector_valued(self):
        y = np.arange(4) * m
        res, _ = quad_vec(lambda x, y=y: x * y, 0 * s, 2 * s)
        # integral of x dx from 0 to 2 s = 2 s**2, times each y
        expected = 2.0 * np.arange(4)
        np.testing.assert_allclose(res.value, expected)
        self.assertEqual(res.dimension, (m * s**2).dimension)

    def test_bound_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            quad_vec(lambda x: x, 0 * m, 1 * s)


class TestDblquad(unittest.TestCase):
    def test_value_and_dimension(self):
        # integral over unit square of 1 -> area 1 m**2
        res, _ = dblquad(lambda y, x: 1, 0 * m, 1 * m, 0 * m, 1 * m)
        self.assertAlmostEqual(res.value, 1.0)
        self.assertEqual(res.dimension, (m**2).dimension)

    def test_x_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            dblquad(lambda y, x: x * y, 0 * m, 1 * s, 0 * m, 1 * m)

    def test_y_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            dblquad(lambda y, x: x * y, 0 * m, 1 * m, 0 * m, 1 * s)


class TestTplquad(unittest.TestCase):
    def test_value_and_dimension(self):
        # integral over unit cube of x*y*z -> (1/2)**3 = 0.125, dim m**6
        res, _ = tplquad(
            lambda z, y, x: x * y * z,
            0 * m,
            1 * m,
            0 * m,
            1 * m,
            0 * m,
            1 * m,
        )
        self.assertAlmostEqual(res.value, 0.125)
        self.assertEqual(res.dimension, (m**6).dimension)

    def test_x_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            tplquad(
                lambda z, y, x: x, 0 * m, 1 * s, 0 * m, 1 * m, 0 * m, 1 * m
            )

    def test_y_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            tplquad(
                lambda z, y, x: x, 0 * m, 1 * m, 0 * m, 1 * s, 0 * m, 1 * m
            )

    def test_z_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            tplquad(
                lambda z, y, x: x, 0 * m, 1 * m, 0 * m, 1 * m, 0 * m, 1 * s
            )


class TestSolveIvp(unittest.TestCase):
    def test_scalar(self):
        # y' = -y, y(0) = 1 m  -> y(t) = exp(-t)
        sol = solve_ivp(
            lambda t, y: -y / (1 * s),
            (0 * s, 1 * s),
            [1 * m],
            t_eval=np.array([0.0, 1.0]) * s,
        )
        self.assertIsInstance(sol.y, Quantity)
        self.assertEqual(sol.y.dimension, m.dimension)
        self.assertEqual(sol.t.dimension, s.dimension)
        self.assertAlmostEqual(sol.y.value[0][0], 1.0, places=3)
        self.assertAlmostEqual(sol.y.value[0][-1], np.exp(-1.0), places=3)

    def test_scalar_dense_output_callable(self):
        sol = solve_ivp(
            lambda t, y: -y / (1 * s),
            (0 * s, 1 * s),
            [1 * m],
            dense_output=True,
        )
        # the wrapped dense solution accepts a Quantity time and is unit-aware
        val = sol.sol(0.5 * s)
        self.assertIsInstance(val, Quantity)
        self.assertEqual(val.dimension, m.dimension)

    def test_non_scalar(self):
        # harmonic oscillator: y0' = y1, y1' = -y0
        def f(t, Y):
            return np.array([Y[1] * (1 / s), -Y[0] / (1 * s)], dtype=object)

        sol = solve_ivp(f, (0 * s, 1 * s), [0 * m, 1 * m])
        self.assertIsInstance(sol.y, list)
        self.assertEqual(len(sol.y), 2)
        for component in sol.y:
            self.assertIsInstance(component, Quantity)
            self.assertEqual(component.dimension, m.dimension)


class TestRoot(unittest.TestCase):
    def test_value_and_dimension(self):
        res = root(lambda x: x - 2 * m, 0.5 * m)
        self.assertIsInstance(res, Quantity)
        self.assertAlmostEqual(res.value, 2.0)
        self.assertEqual(res.dimension, m.dimension)


class TestBrentq(unittest.TestCase):
    def test_value_and_dimension(self):
        res = brentq(lambda x: x - 2 * m, 0 * m, 5 * m)
        self.assertIsInstance(res, Quantity)
        self.assertAlmostEqual(res.value, 2.0)
        self.assertEqual(res.dimension, m.dimension)

    def test_bound_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            brentq(lambda x: x, 0 * m, 1 * s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
