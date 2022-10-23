
import unittest

from physipy import m, Quantity, Dimension, rad
import uncertainties as uc
from uncertainties import umath


import numpy as np

class TestWrappingUncertainties(unittest.TestCase):

    def test_creation_from_mul(self):
        x = uc.ufloat(0.20, 0.01)
        
        x_q_l = x * m
        x_q_r = m * x
        x_q   =  Quantity(x, Dimension('m'))
        
        self.assertEqual(x_q_l, x_q_r)
        self.assertEqual(x_q_l, x_q)
        self.assertEqual(x_q_r, x_q)
        
    def test_mul(self):
        x = uc.ufloat(0.20, 0.01)
        x_q = x*m
        
        res = x_q*m
        exp = Quantity(x, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        res = m*x_q
        exp = Quantity(x, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        res = x_q*x_q
        exp = Quantity(x*x, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        y = uc.ufloat(0.1, 0.02)
        res = x_q*y
        exp = Quantity(x*y, Dimension('m'))
        self.assertEqual(res, exp)
        
        y_q = y*m
        res = x_q*y_q
        exp = Quantity(x*y, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        
    def test_pow(self):
        x = uc.ufloat(0.20, 0.01)
        x_q = x*m
        
        res = x_q**2
        exp = Quantity(x**2, Dimension('m**2'))
        self.assertEqual(res, exp)
        
    def test_sin(self):
        x = uc.ufloat(0.20, 0.01)
        x_q = x*rad
        # umath.sin
        from uncertainties.umath import sin
        
        # expect sin(x_q) to run smoothly as rad is an angle ?
        # and return the same value as raw x
        sin_x = sin(x)
        sin_x_q = sin(x_q)
        
        res = sin_x_q
        exp = sin_x
        self.assertEqual(res, exp)

    def test_array(self):
        x =  uc.ufloat(0.20, 0.01)
        arr = np.array([uc.ufloat(1, 0.01),
                        uc.ufloat(2, 0.1)])

        arr_q =  np.array([arr[0]*m, arr[1]*m])

        self.assertEqual(arr.sum() *m, arr_q.sum())

    def test_nominal(self):
        x = uc.ufloat(0.20, 0.01)
        x_q = x*m
        
        res = x_q.nominal_value
        exp = x.nominal_value * m
        self.assertEqual(res, exp)
        res = x_q.n
        exp = x.n * m
        self.assertEqual(res, exp)
        
    def test_std_dev(self):
        x = uc.ufloat(0.20, 0.01)
        x_q = x*m
        
        res = x_q.std_dev
        exp = x.std_dev * m
        self.assertEqual(res, exp)
        res = x_q.s
        exp = x.s * m
        self.assertEqual(res, exp)


    def test_comp(self):
        x = uc.ufloat(0.20, 0.01)
        y = x + 0.0001
        
        x_q = x*m
        y_q = y*m
        
        res = y_q > x_q
        exp = y   > x
        self.assertEqual(res, exp)
        
        exp = y > 0
        res = y_q > 0*m
        self.assertEqual(res, exp)
        
    def test_eq(self):
        y = uc.ufloat(1, 0.1)
        z = uc.ufloat(1, 0.1)


        y_q = y*m
        z_q = z*m

        res = y_q == y_q
        exp = y   == y
        self.assertEqual(res, exp)
        
        res = z_q == y_q
        exp = z   == y
        self.assertEqual(res, exp)
        
        
    def test_sum(self):
        u = uc.ufloat(1, 0.1, "u variable") 
        v = uc.ufloat(10, 0.1, "v variable")
        u_q = u*m
        v_q = v*m
        
        sum = u+2*v
        sum_value_q = u_q + 2*v_q
        
        res = sum_value_q
        exp = sum*m
        self.assertEqual(res, exp)
        
        res = sum_value_q - (u_q+2*v_q)
        exp = (sum - (u+2*v))*m
        self.assertEqual(res, exp)
        
    def test_umath_sqrt(self):
        x = uc.ufloat(0, 0)
        x_q = x*m

        res = umath.sqrt(x_q)
        exp = umath.sqrt(x)*m**0.5
        self.assertEqual(res, exp)

    def test_neq(self):
        x = uc.ufloat(1, 0.1)
        y = uc.ufloat(3.14, 0.01)
        x_q = x*m
        y_q = y*m
        
        res = x_q != y_q
        exp = x != y
        self.assertEqual(res, exp)

    def test_printing(self):
        x = uc.ufloat(1, 0.1)
        y = uc.ufloat(3.14, 0.01)
        x_q = x*m        
        
        res = 'Result = {:10.2f}'.format(x_q)
        exp = 'Result = {:10.2f}'.format(x) + " m"
        self.assertEqual(res, exp)

        res = '1 significant digit on the uncertainty:  {:.1u}'.format(x_q)
        exp = '1 significant digit on the uncertainty:  {:.1u}'.format(x) + " m"
        self.assertEqual(res, exp)
        
        res = '3 significant digits on the uncertainty: {:.3u}'.format(x_q)
        exp = '3 significant digits on the uncertainty: {:.3u}'.format(x) + " m"
        self.assertEqual(res, exp)

        res = '1 significant digit, exponent notation:  {:.1ue}'.format(x_q)
        exp = '1 significant digit, exponent notation:  {:.1ue}'.format(x) + " m"
        self.assertEqual(res, exp)

        res = '1 significant digit, percentage:         {:.1u%}'.format(x_q)
        exp = '1 significant digit, percentage:         {:.1u%}'.format(x) + " m"
        self.assertEqual(res, exp)

        res = 'Result = {:10.1e}'.format(x_q*1e-10)
        exp = 'Result = {:10.1e}'.format(x*1e-10) + " m"
        self.assertEqual(res, exp)

        
    def test_derivatives(self):
        u = uc.ufloat(1, 0.1) #* m
        v = uc.ufloat(10, 0.1)# * m
        sum_value = u + v
        exp = sum_value.derivatives[u] *m
        
        u_q = u * m
        v_q = v * m
        sum_q = u_q + v_q
        res = sum_q.derivatives[u_q]
        
        self.assertEqual(res, exp)
        
if __name__ == "__main__":
    unittest.main()
        
        
        
                          
                          