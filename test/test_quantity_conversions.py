"""
Unit tests for the Python numeric-conversion protocol on Quantity:
__float__, __int__, __complex__ and the reflected __rfloordiv__.

These dunders are part of the standard numeric protocol (float(q), int(q),
complex(q), `number // q`) and each guards on dimensionlessness, so a
regression would be silently wrong.
"""

import unittest

from physipy import Dimension, DimensionError, Quantity, m, rad, s


class TestFloat(unittest.TestCase):
    def test_dimensionless(self):
        self.assertEqual(float(Quantity(3.5, Dimension(None))), 3.5)

    def test_extended_dimensionless_radians(self):
        # radians are treated as "extended dimensionless" and convert.
        self.assertEqual(float(2.0 * rad), 2.0)

    def test_dimensionful_raises(self):
        with self.assertRaises(DimensionError):
            float(2 * m)


class TestInt(unittest.TestCase):
    def test_dimensionless(self):
        self.assertEqual(int(Quantity(3.9, Dimension(None))), 3)

    def test_extended_dimensionless_radians(self):
        self.assertEqual(int(2.7 * rad), 2)

    def test_dimensionful_raises(self):
        with self.assertRaises(DimensionError):
            int(2 * m)


class TestComplex(unittest.TestCase):
    def test_dimensionless(self):
        self.assertEqual(complex(Quantity(3.5, Dimension(None))), complex(3.5))

    def test_extended_dimensionless_radians(self):
        self.assertEqual(complex(2.0 * rad), complex(2.0))

    def test_dimensionful_raises(self):
        with self.assertRaises(DimensionError):
            complex(2 * m)


class TestRfloordiv(unittest.TestCase):
    """`number // quantity` triggers Quantity.__rfloordiv__."""

    def test_dimensionless_result_is_stripped(self):
        # left operand is a plain number -> treated as dimensionless, and
        # self is dimensionless, so the result is reduced to a plain number.
        res = 7 // Quantity(2.0, Dimension(None))
        self.assertNotIsInstance(res, Quantity)
        self.assertEqual(res, 3.0)

    def test_dimension_mismatch_raises(self):
        # 7 is dimensionless, (2*m) is a length -> mismatch.
        with self.assertRaises(DimensionError):
            7 // (2 * m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
