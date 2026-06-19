"""
Unit tests for physipy.math, the Quantity-aware wrapper around the stdlib
math module.

These tests exercise the *behaviour* of the wrapped functions (value and
dimension of the result, and the dimension-checking branches) rather than
relying only on doctests.
"""

import math
import unittest

import numpy as np

from physipy import Dimension, DimensionError, Quantity, m, rad, s
from physipy import math as pm


class TestOneInSameOut(unittest.TestCase):
    """ceil, floor, trunc, fabs : keep the input dimension."""

    def test_value_and_dimension(self):
        cases = [
            (pm.ceil, 2.3, math.ceil(2.3)),
            (pm.floor, 2.7, math.floor(2.7)),
            (pm.trunc, 2.7, math.trunc(2.7)),
            (pm.fabs, -2.0, math.fabs(-2.0)),
        ]
        for func, raw, expected in cases:
            with self.subTest(func=func):
                res = func(raw * m)
                self.assertIsInstance(res, Quantity)
                self.assertEqual(res.value, expected)
                self.assertEqual(res.dimension, m.dimension)

    def test_accepts_plain_number(self):
        res = pm.ceil(2.3)
        self.assertEqual(res.value, 3)
        self.assertEqual(res.dimension, Dimension(None))


class TestTwoSameInSameOut(unittest.TestCase):
    """fmod, remainder : both args must share a dimension, kept on output."""

    def test_value_and_dimension(self):
        self.assertEqual(pm.fmod(5 * m, 3 * m).value, math.fmod(5, 3))
        self.assertEqual(pm.fmod(5 * m, 3 * m).dimension, m.dimension)
        self.assertEqual(
            pm.remainder(5 * m, 3 * m).value, math.remainder(5, 3)
        )
        self.assertEqual(pm.remainder(5 * m, 3 * m).dimension, m.dimension)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            pm.fmod(5 * m, 3 * s)
        with self.assertRaises(DimensionError):
            pm.remainder(5 * m, 3 * s)


class TestAnyBool(unittest.TestCase):
    """isinf, isfinite, isnan : dimension-agnostic, return plain bool."""

    def test_results(self):
        self.assertTrue(pm.isnan(float("nan") * m))
        self.assertFalse(pm.isnan(1.0 * m))
        self.assertTrue(pm.isinf(float("inf") * m))
        self.assertFalse(pm.isinf(1.0 * m))
        self.assertTrue(pm.isfinite(1.0 * m))
        self.assertFalse(pm.isfinite(float("inf") * m))

    def test_returns_plain_bool(self):
        self.assertIsInstance(pm.isnan(1.0 * m), bool)


class TestAngleOrDimlessToDimless(unittest.TestCase):
    """cos, sin, tan, cosh, sinh, tanh : accept radians or dimensionless."""

    def test_radians_and_dimensionless(self):
        for name in ("cos", "sin", "tan", "cosh", "sinh", "tanh"):
            func = getattr(pm, name)
            ref = getattr(math, name)
            with self.subTest(func=name):
                self.assertAlmostEqual(func(0 * rad), ref(0.0))
                self.assertAlmostEqual(func(0.5), ref(0.5))

    def test_non_angle_dimension_raises(self):
        for name in ("cos", "sin", "tan", "cosh", "sinh", "tanh"):
            with self.subTest(func=name):
                with self.assertRaises(DimensionError):
                    getattr(pm, name)(1 * m)


class TestDimlessToDimless(unittest.TestCase):
    """acos..atanh, erf, exp, log, gamma : require dimensionless input."""

    def test_dimensionless_values(self):
        self.assertAlmostEqual(pm.acos(1.0), math.acos(1.0))
        self.assertAlmostEqual(pm.asin(0.0), math.asin(0.0))
        self.assertAlmostEqual(pm.atan(1.0), math.atan(1.0))
        self.assertAlmostEqual(pm.acosh(1.0), math.acosh(1.0))
        self.assertAlmostEqual(pm.asinh(0.0), math.asinh(0.0))
        self.assertAlmostEqual(pm.atanh(0.0), math.atanh(0.0))
        self.assertAlmostEqual(pm.erf(1.0), math.erf(1.0))
        self.assertAlmostEqual(pm.erfc(1.0), math.erfc(1.0))
        self.assertAlmostEqual(pm.gamma(1.0), math.gamma(1.0))
        self.assertAlmostEqual(pm.lgamma(2.0), math.lgamma(2.0))
        self.assertAlmostEqual(pm.exp(1.0), math.exp(1.0))
        self.assertAlmostEqual(pm.log(1.0), math.log(1.0))
        self.assertAlmostEqual(pm.log10(10.0), math.log10(10.0))
        self.assertAlmostEqual(pm.log2(8.0), math.log2(8.0))
        self.assertAlmostEqual(pm.log1p(0.5), math.log1p(0.5))
        self.assertAlmostEqual(pm.expm1(0.5), math.expm1(0.5))

    def test_dimensionful_input_raises(self):
        for name in ("acos", "exp", "log", "erf", "gamma"):
            with self.subTest(func=name):
                with self.assertRaises(DimensionError):
                    getattr(pm, name)(1 * m)


class TestTwoSameToDimless(unittest.TestCase):
    """atan2 : returns a plain (dimensionless) float."""

    def test_value(self):
        self.assertAlmostEqual(pm.atan2(1 * m, 1 * m), math.atan2(1, 1))


class TestCopysign(unittest.TestCase):
    def test_value_and_dimension(self):
        res = pm.copysign(3 * m, -1)
        self.assertEqual(res.value, -3.0)
        self.assertEqual(res.dimension, m.dimension)


class TestSqrt(unittest.TestCase):
    def test_value_and_dimension(self):
        res = pm.sqrt(4 * m**2)
        self.assertAlmostEqual(res.value, 2.0)
        self.assertEqual(res.dimension, (m**2).dimension ** 0.5)
        self.assertEqual(res.dimension, m.dimension)


class TestAnyToSame(unittest.TestCase):
    """fsum : keeps dimension, operates on an array-valued Quantity."""

    def test_value_and_dimension(self):
        res = pm.fsum(np.array([1.0, 2.0, 3.0]) * m)
        self.assertEqual(res.value, 6.0)
        self.assertEqual(res.dimension, m.dimension)


class TestNotImplemented(unittest.TestCase):
    """degrees, radians, gcd, ... are explicitly not implemented."""

    def test_raise_not_implemented(self):
        for name in (
            "degrees",
            "radians",
            "gcd",
            "isclose",
            "isqrt",
            "ldexp",
            "modf",
            "perm",
            "pow",
        ):
            with self.subTest(func=name):
                with self.assertRaises(NotImplementedError):
                    getattr(pm, name)(1, 2)


class TestMute(unittest.TestCase):
    """dist, hypot, prod : passed through to stdlib math unchanged."""

    def test_passthrough(self):
        self.assertAlmostEqual(pm.dist([1, 2], [3, 4]), math.dist([1, 2], [3, 4]))
        self.assertAlmostEqual(pm.hypot(3, 4), math.hypot(3, 4))
        self.assertEqual(pm.prod([2, 3]), math.prod([2, 3]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
