import unittest
from fractions import Fraction

from physipy import Dimension, DimensionError


class TestClassDimension(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.m = Dimension("L")
        cls.none = Dimension(None)
        cls.dim_complexe = Dimension({"J": 1, "theta": -3})
        cls.no_dimension_str = "no-dimension"

    def test_010_init(cls):

        metre_by_dict = Dimension({"L": 1})
        cls.assertEqual(cls.m, metre_by_dict)

        none_dimenion_dict = cls.none.dim_dict
        dico_dimension_none = {'L': 0,
                               'M': 0,
                               'T': 0,
                               'I': 0,
                               'theta': 0,
                               'N': 0,
                               'J': 0,
                               'RAD': 0,
                               'SR': 0}
        cls.assertEqual(none_dimenion_dict, dico_dimension_none)

        cls.assertRaises(TypeError, lambda: Dimension({"m": 1}))

    def test_020_str(cls):

        expected_str = "L"
        actual_str = str(cls.m)
        cls.assertEqual(expected_str, actual_str)

        expected_str = "J/theta**3"
        actual_str = str(cls.dim_complexe)
        cls.assertEqual(expected_str, actual_str)

        expected_str = cls.no_dimension_str
        actual_str = str(cls.none)
        cls.assertEqual(expected_str, actual_str)

    def test_030_repr(cls):

        cls.assertEqual(repr(cls.none), "<Dimension : {'L': 0, 'M': 0, 'T': 0, 'I': 0, 'theta': 0, 'N': 0, 'J': 0, 'RAD': 0, 'SR': 0}>")
        cls.assertEqual(repr(cls.m), "<Dimension : {'L': 1, 'M': 0, 'T': 0, 'I': 0, 'theta': 0, 'N': 0, 'J': 0, 'RAD': 0, 'SR': 0}>")
        cls.assertEqual(repr(cls.dim_complexe), "<Dimension : {'L': 0, 'M': 0, 'T': 0, 'I': 0, 'theta': -3, 'N': 0, 'J': 1, 'RAD': 0, 'SR': 0}>")

    def test_040_mul(cls):

        cls.assertEqual(cls.m * cls.dim_complexe,
                        Dimension({"J": 1, "L": 1, "theta": -3}))

        # Multipliying by a number, not a Dimension object
        cls.assertRaises(TypeError, lambda: cls.m * 1.12)
        cls.assertRaises(TypeError, lambda: 1.12 * cls.m)

    def test_050_div(cls):

        cls.assertEqual(cls.m / cls.dim_complexe,
                        Dimension({"J": -1, "L": 1, "theta": 3}))
        # Testing the inversion by dividing 1
        cls.assertEqual(1 / cls.m,
                        Dimension({"L": -1}))

        # Dividing by a number, not a Dimension object
        cls.assertRaises(TypeError, lambda: cls.m / 1.12)
        cls.assertRaises(TypeError, lambda: 1.12 / cls.m)
        
        #cls.assertEqual(cls.m/1,
        #               cls.m)

    def test_060_pow(cls):

        cls.assertEqual(cls.m ** 2, Dimension({"L": 2}))
        cls.assertEqual(cls.m ** (1/2), Dimension({"L": 1/2}))
        cls.assertEqual(cls.m ** 1.2, Dimension({"L": 1.2}))
        cls.assertEqual(cls.m ** Fraction(1/2), Dimension({"L":Fraction(1/2)}))
        
        # complex
        #cls.assertRaises(TypeError, lambda: cls.m ** 1.2j)

    def test_070_eq_ne(cls):

        cls.assertTrue(cls.m == Dimension({"L": 1}))
        cls.assertTrue(cls.m != cls.none)

    #def test_080_inverse(cls):
    #    m_inverse = cls.m.inverse()
    #    cls.assertEqual(m_inverse, Dimension({"L": -1}))
    def test_080_pow_inverse(cls):
        m_inverse = 1/cls.m
        cls.assertEqual(m_inverse, Dimension({"L":-1}))
        
        
    def test_090_str_SI_unit(cls):
        cls.assertEqual(cls.m.str_SI_unit(), "m")
        cls.assertEqual(cls.none.str_SI_unit(),"")

    def test_100_expr_parsing(cls):
        cls.assertEqual(cls.m, Dimension("L"))
        cls.assertEqual(cls.m, Dimension("L**1"))
        cls.assertEqual(cls.m * cls.m, Dimension("L**2"))
        cls.assertEqual(cls.m * cls.dim_complexe, Dimension("L*J/theta**3"))
        
        cls.assertEqual(cls.m, Dimension("m"))
        cls.assertEqual(cls.m * cls.m, Dimension("m**2"))
        cls.assertEqual(cls.m * cls.dim_complexe, Dimension("m*cd/K**3"))
        
        with cls.assertRaises(AttributeError):
            # sympy parsing not good with ^ char
            cls.assertEqual(cls.m * cls.m, Dimension("m^2"))
            
    def test_101_dimensionality(cls):
        cls.assertEqual(cls.m.dimensionality, 'length')
    
    
    def test_110_siunit_dict(cls):
        cls.assertEqual(Dimension(None).siunit_dict(), 
                        {'m': 0, 
                         'kg': 0, 
                         's': 0, 
                         'A': 0, 
                         'K': 0, 
                         'mol': 0, 
                         'cd': 0, 
                         'rad': 0, 
                         'sr': 0})
        cls.assertEqual(Dimension({"L":1}).siunit_dict(), 
                        {'m': 1, 
                         'kg': 0, 
                         's': 0, 
                         'A': 0, 
                         'K': 0, 
                         'mol': 0, 
                         'cd': 0, 
                         'rad': 0, 
                         'sr': 0})
        cls.assertEqual(Dimension({"L":1.2}).siunit_dict(), 
                        {'m': 1.2, 
                         'kg': 0, 
                         's': 0, 
                         'A': 0, 
                         'K': 0, 
                         'mol': 0, 
                         'cd': 0, 
                         'rad': 0, 
                         'sr': 0})
        cls.assertEqual(Dimension({"L":1/2}).siunit_dict(), 
                        {'m': 1/2, 
                         'kg': 0, 
                         's': 0, 
                         'A': 0, 
                         'K': 0, 
                         'mol': 0, 
                         'cd': 0, 
                         'rad': 0, 
                         'sr': 0})
        
        cls.assertEqual(Dimension({"J": -1, 
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
        
        
    def test_repr_latex(cls):
        cls.assertEqual(Dimension(None)._repr_latex_(),
                       "$1$")
        cls.assertEqual(Dimension({"L":1})._repr_latex_(),
                       "$L$")
        cls.assertEqual(Dimension({"L":2})._repr_latex_(),
                       "$L^{2}$")
        cls.assertEqual(Dimension({"J": -1, 
                                   "L": 1, 
                                   "theta": 3})._repr_latex_(),
                       r"$\frac{L \theta^{3}}{J}$")


    def test_latex_SI_unit(cls):
        cls.assertEqual(Dimension(None).latex_SI_unit(),
                       "$1$")
        cls.assertEqual(Dimension({"L":1}).latex_SI_unit(),
                       "$m$")
        cls.assertEqual(Dimension({"L":2}).latex_SI_unit(),
                       "$m^{2}$")
        cls.assertEqual(Dimension({"J": -1, 
                                   "L": 1, 
                                   "theta": 3}).latex_SI_unit(),
                        r"$\frac{K^{3} m}{cd}$")
        
        
    
    #def test_pycodestyle(cls):
    #    import pycodestyle
    #    style = pycodestyle.StyleGuide(quiet=True)
    #    result = style.check_files(['dimension.py', 'test_dimension.py'])
    #    cls.assertEqual(result.total_errors, 0,
    #                    "Found code style errors (and warnings).")

if __name__ == "__main__":
    unittest.main()