import unittest
from fractions import Fraction
import time

from physipy import Dimension, DimensionError
from physipy.quantity import dimension

import pandas as pd

import doctest
import unittest

# The load_tests() function is automatically called by unittest
# see https://docs.python.org/3/library/doctest.html#unittest-api
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(dimension))
    return tests


class TestDimension(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        self.tottime = 0

    def tearDown(self):
        t = time.time() - self.startTime
        self.tottime = self.tottime + t
        print(f"{self.id():70} : {t:10.6f}")
        self.times.append(t)
        self.ids.append(str(self.id()))

    @classmethod
    def setUpClass(cls):
        cls.m = Dimension("L")
        cls.none = Dimension(None)
        cls.dim_complexe = Dimension({"J": 1, "theta": -3})
        cls.no_dimension_str = "no-dimension"
        cls.times = []
        cls.ids = []
        cls.amp = Dimension("I")

    @classmethod
    def tearDownClass(cls):
        cls.df = pd.DataFrame.from_dict({
            "time": cls.times,
            "id": cls.ids,
        })

    def test_010_init(self):

        metre_by_dict = Dimension({"L": 1})
        self.assertEqual(self.m, metre_by_dict)

        none_dimenion_dict = self.none.dim_dict
        dico_dimension_none = {'L': 0,
                               'M': 0,
                               'T': 0,
                               'I': 0,
                               'theta': 0,
                               'N': 0,
                               'J': 0,
                               'RAD': 0,
                               'SR': 0}
        self.assertEqual(none_dimenion_dict, dico_dimension_none)

        self.assertRaises(TypeError, lambda: Dimension({"m": 1}))

    def test_020_str(self):

        expected_str = "L"
        actual_str = str(self.m)
        self.assertEqual(expected_str, actual_str)

        expected_str = "J/theta**3"
        actual_str = str(self.dim_complexe)
        self.assertEqual(expected_str, actual_str)

        expected_str = self.no_dimension_str
        actual_str = str(self.none)
        self.assertEqual(expected_str, actual_str)

    def test_030_repr(self):

        self.assertEqual(repr(
            self.none), "<Dimension : {'L': 0, 'M': 0, 'T': 0, 'I': 0, 'theta': 0, 'N': 0, 'J': 0, 'RAD': 0, 'SR': 0}>")
        self.assertEqual(repr(
            self.m), "<Dimension : {'L': 1, 'M': 0, 'T': 0, 'I': 0, 'theta': 0, 'N': 0, 'J': 0, 'RAD': 0, 'SR': 0}>")
        self.assertEqual(repr(self.dim_complexe),
                        "<Dimension : {'L': 0, 'M': 0, 'T': 0, 'I': 0, 'theta': -3, 'N': 0, 'J': 1, 'RAD': 0, 'SR': 0}>")

    def test_040_mul(self):

        self.assertEqual(self.m * self.dim_complexe,
                        Dimension({"J": 1, "L": 1, "theta": -3}))

        # Multipliying by a number, not a Dimension object
        self.assertRaises(TypeError, lambda: self.m * 1.12)
        self.assertRaises(TypeError, lambda: 1.12 * self.m)

    def test_050_div(self):

        self.assertEqual(self.m / self.dim_complexe,
                        Dimension({"J": -1, "L": 1, "theta": 3}))
        # Testing the inversion by dividing 1
        self.assertEqual(1 / self.m,
                        Dimension({"L": -1}))

        # Dividing by a number, not a Dimension object
        self.assertRaises(TypeError, lambda: self.m / 1.12)
        self.assertRaises(TypeError, lambda: 1.12 / self.m)

        # self.assertEqual(self.m/1,
        #               self.m)

    def test_060_pow(self):

        self.assertEqual(self.m ** 2, Dimension({"L": 2}))
        self.assertEqual(self.m ** (1/2), Dimension({"L": 1/2}))
        self.assertEqual(self.m ** 1.2, Dimension({"L": 1.2}))
        self.assertEqual(self.m ** Fraction(1/2),
                        Dimension({"L": Fraction(1/2)}))

        # complex
        #self.assertRaises(TypeError, lambda: self.m ** 1.2j)

    def test_070_eq_ne(self):

        self.assertTrue(self.m == Dimension({"L": 1}))
        self.assertTrue(self.m != self.none)

    # def test_080_inverse(self):
    #    m_inverse = self.m.inverse()
    #    self.assertEqual(m_inverse, Dimension({"L": -1}))
    def test_080_pow_inverse(self):
        m_inverse = 1/self.m
        self.assertEqual(m_inverse, Dimension({"L": -1}))

    def test_090_str_SI_unit(self):
        self.assertEqual(self.m.str_SI_unit(), "m")
        self.assertEqual(self.none.str_SI_unit(), "")

    def test_100_expr_parsing(self):
        self.assertEqual(self.m, Dimension("L"))
        self.assertEqual(self.m, Dimension("L**1"))
        self.assertEqual(self.m * self.m, Dimension("L**2"))
        self.assertEqual(self.m * self.dim_complexe, Dimension("L*J/theta**3"))

        self.assertEqual(self.m, Dimension("m"))
        self.assertEqual(self.m * self.m, Dimension("m**2"))
        self.assertEqual(self.m * self.dim_complexe, Dimension("m*cd/K**3"))

        # sympy was parsing "I" as complex number
        self.assertEqual(self.amp*self.m, Dimension("L*I"))

        with self.assertRaises(TypeError):
            # sympy parsing not good with ^ char
            Dimension("m^2")

    def test_101_dimensionality(self):
        self.assertEqual(self.m.dimensionality, 'length')

    def test_110_siunit_dict(self):
        self.assertEqual(Dimension(None).siunit_dict(),
                        {'m': 0,
                         'kg': 0,
                         's': 0,
                         'A': 0,
                         'K': 0,
                         'mol': 0,
                         'cd': 0,
                         'rad': 0,
                         'sr': 0})
        self.assertEqual(Dimension({"L": 1}).siunit_dict(),
                        {'m': 1,
                         'kg': 0,
                         's': 0,
                         'A': 0,
                         'K': 0,
                         'mol': 0,
                         'cd': 0,
                         'rad': 0,
                         'sr': 0})
        self.assertEqual(Dimension({"L": 1.2}).siunit_dict(),
                        {'m': 1.2,
                         'kg': 0,
                         's': 0,
                         'A': 0,
                         'K': 0,
                         'mol': 0,
                         'cd': 0,
                         'rad': 0,
                         'sr': 0})
        self.assertEqual(Dimension({"L": 1/2}).siunit_dict(),
                        {'m': 1/2,
                         'kg': 0,
                         's': 0,
                         'A': 0,
                         'K': 0,
                         'mol': 0,
                         'cd': 0,
                         'rad': 0,
                         'sr': 0})

        self.assertEqual(Dimension({"J": -1,
                                   "L": 1,
                                   "theta": 3}).siunit_dict(),
                        {'m': 1,
                         'kg': 0,
                         's': 0,
                         'A': 0,
                         'K': 3,
                         'mol': 0,
                         'cd': -1,
                         'rad': 0,
                         'sr': 0}
                        )

    def test_repr_latex(self):
        self.assertEqual(Dimension(None)._repr_latex_(),
                        "$1$")
        self.assertEqual(Dimension({"L": 1})._repr_latex_(),
                        "$L$")
        self.assertEqual(Dimension({"L": 2})._repr_latex_(),
                        "$L^{2}$")
        self.assertEqual(Dimension({"J": -1,
                                   "L": 1,
                                   "theta": 3})._repr_latex_(),
                        r"$\frac{L \theta^{3}}{J}$")

    def test_latex_SI_unit(self):
        self.assertEqual(Dimension(None).latex_SI_unit(),
                        "$1$")
        self.assertEqual(Dimension({"L": 1}).latex_SI_unit(),
                        "$m$")
        self.assertEqual(Dimension({"L": 2}).latex_SI_unit(),
                        "$m^{2}$")
        self.assertEqual(Dimension({"J": -1,
                                   "L": 1,
                                   "theta": 3}).latex_SI_unit(),
                        r"$\frac{K^{3} m}{cd}$")

    # def test_pycodestyle(self):
    #    import pycodestyle
    #    style = pycodestyle.StyleGuide(quiet=True)
    #    result = style.check_files(['dimension.py', 'test_dimension.py'])
    #    self.assertEqual(result.total_errors, 0,
    #                    "Found code style errors (and warnings).")
if __name__ == "__main__":
    unittest.main()
