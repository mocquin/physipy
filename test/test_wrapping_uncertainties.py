
import unittest

from physipy import m, Quantity, Dimension, rad
import uncertainties as uc
from uncertainties import umath
import pandas as pd
import time

import numpy as np

class TestWrappingUncertainties(unittest.TestCase):
    
    
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
        
        cls.x = uc.ufloat(0.20, 0.01)
        cls.x_q = cls.x * m
        cls.y = uc.ufloat(0.20, 0.01)
        cls.y_q = cls.y * m
        cls.z = uc.ufloat(3, 0.01)
        cls.z_q = cls.z * m        
    
        cls.times = []
        cls.ids = []

    @classmethod
    def tearDownClass(cls):
        cls.df = pd.DataFrame.from_dict({
            "time":cls.times,
            "id":cls.ids,
        }) 

    def test_creation_from_mul(self):
        
        x_q_l = self.x * m
        x_q_r = m * self.x
        x_q   =  Quantity(self.x, Dimension('m'))
        
        self.assertEqual(x_q_l, x_q_r)
        self.assertEqual(x_q_l, x_q)
        self.assertEqual(x_q_r, x_q)
        
    def test_mul(self):
        
        res = self.x_q*m
        exp = Quantity(self.x, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        res = m*self.x_q
        exp = Quantity(self.x, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        res = self.x_q*self.x_q
        exp = Quantity(self.x*self.x, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        res = self.x_q*self.y
        exp = Quantity(self.x*self.y, Dimension('m'))
        self.assertEqual(res, exp)
        
        res = self.x_q*self.y_q
        exp = Quantity(self.x*self.y, Dimension('m**2'))
        self.assertEqual(res, exp)
        
        
    def test_pow(self):
        x_q = self.x*m
        
        res = self.x_q**2
        exp = Quantity(self.x**2, Dimension('m**2'))
        self.assertEqual(res, exp)
        
    def test_sin(self):
        x_q = self.x*rad
        # umath.sin
        from uncertainties.umath import sin
        
        # expect sin(x_q) to run smoothly as rad is an angle ?
        # and return the same value as raw x
        sin_x = sin(self.x)
        sin_x_q = sin(x_q)
        
        res = sin_x_q
        exp = sin_x
        self.assertEqual(res, exp)

    def test_array_sum(self):
        arr = np.array([self.x,
                        self.y])

        arr_q =  np.array([arr[0]*m, arr[1]*m])

        self.assertEqual(arr.sum() *m, arr_q.sum())

    def test_nominal(self):        
        res = self.x_q.nominal_value
        exp = self.x.nominal_value * m
        self.assertEqual(res, exp)
        res = self.x_q.n
        exp = self.x.n * m
        self.assertEqual(res, exp)
        
    def test_std_dev(self):
        
        res = self.x_q.std_dev
        exp = self.x.std_dev * m
        self.assertEqual(res, exp)
        res = self.x_q.s
        exp = self.x.s * m
        self.assertEqual(res, exp)


    def test_comp(self):
        res = self.y_q > self.x_q
        exp = self.y   > self.x
        self.assertEqual(res, exp)
        
        exp = self.y   > 0
        res = self.y_q > 0*m
        self.assertEqual(res, exp)
        
    def test_eq(self):

        res = self.y_q == self.y_q
        exp = self.y   == self.y
        self.assertEqual(res, exp)
        
        res = self.z_q == self.y_q
        exp = self.z   == self.y
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
        
        res = umath.sqrt(self.x_q)
        exp = umath.sqrt(self.x)*m**0.5
        self.assertEqual(res, exp)

    def test_neq(self):
        
        res = self.x_q != self.y_q
        exp = self.x   != self.y
        self.assertEqual(res, exp)

    def test_printing(self):
        
        res = 'Result = {:10.2f}'.format(self.x_q)
        exp = 'Result = {:10.2f}'.format(self.x) + " m"
        self.assertEqual(res, exp)

        res = '1 significant digit on the uncertainty:  {:.1u}'.format(self.x_q)
        exp = '1 significant digit on the uncertainty:  {:.1u}'.format(self.x) + " m"
        self.assertEqual(res, exp)
        
        res = '3 significant digits on the uncertainty: {:.3u}'.format(self.x_q)
        exp = '3 significant digits on the uncertainty: {:.3u}'.format(self.x) + " m"
        self.assertEqual(res, exp)

        res = '1 significant digit, exponent notation:  {:.1ue}'.format(self.x_q)
        exp = '1 significant digit, exponent notation:  {:.1ue}'.format(self.x) + " m"
        self.assertEqual(res, exp)

        res = '1 significant digit, percentage:         {:.1u%}'.format(self.x_q)
        exp = '1 significant digit, percentage:         {:.1u%}'.format(self.x) + " m"
        self.assertEqual(res, exp)

        res = 'Result = {:10.1e}'.format(self.x_q*1e-10)
        exp = 'Result = {:10.1e}'.format(self.x*1e-10) + " m"
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
        
        
        
                          
                          