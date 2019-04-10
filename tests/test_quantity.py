###############################################################################
import numpy as np
import unittest

from physipy.quantity import Dimension, Quantity, DimensionError
#from quantity import DISPLAY_DIGITS, EXP_THRESHOLD
from physipy.quantity import interp, vectorize, integrate_trapz, linspace, quad, dblquad, tplquad #turn_scalar_to_str
from physipy.quantity import SI_units, units#, custom_units
from physipy.quantity import m, s, kg, A, cd, K, mol
from physipy.quantity import quantify, make_quantity


# from quantity import SI_units as u

#m = SI_units["m"]


class TestQuantity(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.x = 12.1111
        cls.y = np.array([1,2,3,4.56])
        cls.z = np.array([1,1,1])
        cls.dim = Dimension("L")
        cls.mm = Quantity(0.001, cls.dim, symbol="mm")
        
        cls.x_q = Quantity(cls.x, cls.dim)
        cls.y_q = Quantity(cls.y, cls.dim)
        cls.z_q = Quantity(cls.z, cls.dim)
        cls.x_qs = Quantity(cls.x, cls.dim, symbol="x_qs")
        cls.y_qs = Quantity(cls.y, cls.dim, symbol="y_qs")
        cls.x_qu = Quantity(cls.x, cls.dim, favunit=cls.mm)
        cls.y_qu = Quantity(cls.y, cls.dim, favunit=cls.mm)
        cls.x_qsu = Quantity(cls.x, cls.dim, symbol="x_qsu", favunit=cls.mm)
        cls.y_qsu = Quantity(cls.y, cls.dim, symbol="y_qsu", favunit=cls.mm)
        
    def test_10_test_produit_division(self):
        self.assertEqual(2 * self.x_q, Quantity(2 * self.x, self.dim))
        self.assertEqual(self.x_q * 2, Quantity(self.x * 2, self.dim))
        self.assertEqual(2 / self.x_q, Quantity(2 / self.x, 1/ self.dim))
        self.assertEqual(self.x_q / 2, Quantity(self.x / 2, self.dim))
        
    def test_20_test_inv(self):
        pass
        
    def test_30_test_unite_favorite(self):
        
        
        #print("DISPLAY_DIGITS : " + str(DISPLAY_DIGITS))
        #print("EXP_THRESHOLD : "+ str(EXP_THRESHOLD))
        longueur = 543.21*m
        longueur.favunit = self.mm
        self.assertEqual(str(longueur),"543210.0 mm")
        
        cy = Quantity(1,Dimension(None),symbol="cy")
        f_cymm = 5*cy/self.mm
        f_cymm.favunit = cy/self.mm
        self.assertEqual(str(f_cymm),"5.0 cy/mm")
        
        
    
    def test_40_interpolateur(self):
        # liste réels interpole liste réels
        tab_x = [1, 2,3,4,5,6,7,8,9,10]
        tab_y = [10,9,8,7,6,5,4,3,2,1]
        variable = 4.5
        self.assertEqual(interp(variable, tab_x, tab_y),6.5)
        
        # array interpole array
        tab_x = np.array([1, 2,3,4,5,6,7,8,9,10])
        tab_y = np.array([10,9,8,7,6,5,4,3,2,1])
        variable = 4.5
        self.assertEqual(interp(variable, tab_x, tab_y),6.5)
        # Quantité interpole array
        m = Quantity(1,Dimension("L"),symbol="m")
        tab_x = np.array([1, 2,3,4,5,6,7,8,9,10])*m
        tab_y = np.array([10,9,8,7,6,5,4,3,2,1])
        variable = 4.5*m
        self.assertEqual(interp(variable, tab_x, tab_y),6.5)
        # Quantite interpole quantite
        m = Quantity(1,Dimension("L"),symbol="m")
        tab_x = np.array([1, 2,3,4,5,6,7,8,9,10])*m
        tab_y = np.array([10,9,8,7,6,5,4,3,2,1])*m**2
        variable = 4.5*m
        self.assertEqual(interp(variable, tab_x, tab_y),6.5*m**2)
        # Array interpole quantite
        m = Quantity(1,Dimension("L"),symbol="m")
        tab_x = np.array([1, 2,3,4,5,6,7,8,9,10])
        tab_y = np.array([10,9,8,7,6,5,4,3,2,1])*m
        variable = 4.5
        self.assertEqual(interp(variable, tab_x, tab_y),6.5*m)
        # Interpole quantité par réel
        with self.assertRaises(DimensionError):
            m = Quantity(1,Dimension("L"),symbol="m")
            tab_x = np.array([1, 2,3,4,5,6,7,8,9,10])*m
            tab_y = np.array([10,9,8,7,6,5,4,3,2,1])
            variable = 4.5
            interp(variable, tab_x, tab_y)
            
        # Interpole array par quantité
        with self.assertRaises(DimensionError):
            m = Quantity(1,Dimension("L"),symbol="m")
            tab_x = np.array([1, 2,3,4,5,6,7,8,9,10])
            tab_y = np.array([10,9,8,7,6,5,4,3,2,1])
            variable = 4.5*m
            interp(variable, tab_x, tab_y)
            
        # Interpole Quantité avec mauvais dimension
        with self.assertRaises(DimensionError):
            m = Quantity(1,Dimension("L"),symbol="m")
            tab_x = np.array([1,2,3,4,5,6,7,8,9,10])*Quantity(1,
                                                              Dimension("J"),
                                                              symbol="cd")
            tab_y = np.array([10,9,8,7,6,5,4,3,2,1])
            variable = 4.5*m
            interp(variable, tab_x, tab_y)

    def test_50_iterateur(self):
        u = np.array([1,2,3,4,5])*m
        u.unite_favorite = Quantity(0.001,Dimension("L"),symbol="mm")
        self.assertEqual(u[2],3*m)
        
        with self.assertRaises(DimensionError):
            u[4] = 35*Quantity(1,Dimension("M"))  
            
    def test_60_add_sub(self):
        with self.assertRaises(DimensionError):
            self.x_q + 1
        self.assertEqual(self.x_q + self.x_q, Quantity(self.x*2,Dimension("L")))
        self.assertEqual(self.x_q - self.x_q, Quantity(0, Dimension("L")))
        
    
    def test_70_mul(self):
        ## Scalaire
        # Q * 2 et 2 * Q
        self.assertEqual(self.x_q * 2,
                        Quantity(self.x_q.value * 2, self.x_q.dimension))
        self.assertEqual(2 * self.x_q ,
                        Quantity(2 * self.x_q.value, self.x_q.dimension))   
        # Q * Q
        self.assertEqual(self.x_q * self.x_q,
                        Quantity(self.x_q.value**2, self.x_q.dimension**2))
        ## array
        # Q * 2 et 2 * Q
        self.assertTrue(np.all(self.y_q * 2 == Quantity(self.y_q.value * 2, self.y_q.dimension)))
        self.assertTrue(np.all(2 * self.y_q == Quantity(2 * self.y_q.value, self.y_q.dimension)))   
        # Q * Q
        self.assertTrue(np.all(self.y_q * self.y_q == Quantity(self.y_q.value**2, self.y_q.dimension**2)))
        
        ## Scalaire et array
        self.assertTrue(np.all(self.x_q * self.y_q == Quantity(self.x_q.value * self.y_q.value,
                                 self.x_q.dimension * self.y_q.dimension)))
        self.assertTrue(np.all(self.y_q * self.x_q == Quantity(self.y_q.value * self.x_q.value,
                                 self.y_q.dimension * self.x_q.dimension)))

        
    
    def test_80_pow(self):
        # x_q
        self.assertEqual(self.x_q ** 1, 
                         self.x_q)
        self.assertEqual(self.x_q ** 2, 
                         Quantity(self.x_q.value**2,self.x_q.dimension**2))
        self.assertEqual(self.x_q ** (self.x_q/self.x_q),
                        self.x_q)
        # y_q
        self.assertTrue(np.all(self.y_q ** 1 == self.y_q))
        self.assertTrue(np.all(self.y_q ** 2 == Quantity(self.y_q.value**2,self.y_q.dimension**2)))
        with self.assertRaises(TypeError):
            # can't power by an array : the resulting dimension's
            # will not be consistents
            # this could work because the resulting power is an a
            # rray of 1, but it won't always be. Consider powering 
            # 1 meter by [2 3 4].
            self.y_q ** (self.y_q/self.y_q) == self.y_q
        # z_q
        self.assertTrue(np.all(self.z_q ** 1 == self.z_q))
        self.assertTrue(np.all(self.z_q ** 2 == Quantity(self.z_q.value**2,self.z_q.dimension**2)))
        with self.assertRaises(TypeError):
            # see above
            self.z_q ** (self.z_q/self.z_q) == self.z_q
        
        # x_qs
        self.assertEqual(self.x_qs ** 1, 
                         self.x_qs)
        self.assertEqual(self.x_qs ** 2, 
                         Quantity(self.x_qs.value**2,
                                  self.x_qs.dimension**2))
        self.assertEqual(self.x_qs ** (self.x_qs/self.x_qs),
                         self.x_qs)
        
        # y_qs
        self.assertTrue(np.all(self.y_qs ** 1 == self.y_qs))
        self.assertTrue(np.all(self.y_qs ** 2 == Quantity(self.y_qs.value**2,self.y_qs.dimension**2)))
        with self.assertRaises(TypeError):
            self.y_qs ** (self.y_qs/self.y_qs) == self.y_qs
        
        # x_qu
        self.assertEqual(self.x_qu ** 1, 
                         self.x_qu)
        self.assertEqual(self.x_qu ** 2, 
                         Quantity(self.x_qu.value**2,
                                  self.x_qu.dimension**2))
        self.assertEqual(self.x_qu ** (self.x_qu/self.x_qu),
                         self.x_qu)
        
        # y_qu
        self.assertTrue(np.all(self.y_qu ** 1 == self.y_qu))
        self.assertTrue(np.all(self.y_qu ** 2 == Quantity(self.y_qu.value**2,self.y_qu.dimension**2)))
        with self.assertRaises(TypeError):
            self.y_qu ** (self.y_qu/self.y_qu) == self.y_qu
        
        # x_qsu
        self.assertEqual(self.x_qsu ** 1, 
                         self.x_qsu)
        self.assertEqual(self.x_qsu ** 2, 
                         Quantity(self.x_qsu.value**2,
                                  self.x_qsu.dimension**2))
        self.assertEqual(self.x_qsu ** (self.x_qsu/self.x_qsu),
                         self.x_qsu)
        
        # y_qsu
        self.assertTrue(np.all(self.y_qsu ** 1 == self.y_qsu))
        self.assertTrue(np.all(self.y_qsu ** 2 == Quantity(self.y_qsu.value**2,self.y_qsu.dimension**2)))
        with self.assertRaises(TypeError):
            self.y_qsu ** (self.y_qsu/self.y_qsu) == self.y_qsu       
    
    
    def test_90_getteur(self):
        # Sans unité favorite
        self.assertEqual(str(self.y_q[2]),"3.0 m")
        # Avec unité favorite
        self.assertEqual(str(self.y_qu[2]),"3000.0 mm")
        
    def test_100_vectorizateur(self):
        mm = Quantity(0.001,Dimension("L"),symbol="mm")
        f_cymm = 2/mm
        f_cymm_array = np.array([1,2,3,4,5])/mm
        @vectorize
        def calcul_FTM_test(fcymm):
            if fcymm > 1/Quantity(1,Dimension("L")):
                return 1-fcymm/(10*1/mm)
            else:
                return 0
        self.assertEqual(calcul_FTM_test(f_cymm),0.8)
        self.assertEqual(list(calcul_FTM_test(f_cymm_array)),
                         list(np.array([0.90,0.80,0.70,0.60,0.50])))
        
    def test_110_integrate_trapz(self):
        mum = Quantity(0.000001,Dimension("L"), symbol="mum")
        l_min = 1 * mum
        l_max = 2 * mum
        def Q_func_1(x):return 1
        self.assertEqual(integrate_trapz(l_min, l_max, Q_func_1),
                         1 * mum)
        def Q_func_x(x): return x
        self.assertEqual(integrate_trapz(l_min, l_max, Q_func_x),
                        1.5 * mum**2)
        # TODO : xmin = xmax
        
    def test_120_linspace(self):
        
        m = Quantity(1,Dimension("L"))
        self.assertEqual(str(linspace(1*m, 2*m, 8)),
                        '[1.   1.14 1.29 1.43 1.57 1.71 1.86 2.  ] m')
        with self.assertRaises(DimensionError):
            linspace(1*m, 2)
    
    def test_130_sum(self):
        #self.assertEqual(sum(cls.x_q), 
        #self.assertEqual(sum(cls.y_q), 
        ##self.assertEqual(sum(cls.x_qs), 
        #self.assertEqual(sum(cls.y_qs), 
        #self.assertEqual(sum(cls.x_qu),
        #self.assertEqual(sum(cls.y_qu), 
        #self.assertEqual(sum(cls.x_qsu), 
        #self.assertEqual(sum(cls.y_qsu)
        pass
        
    def test_140_integrate(self):
        self.assertEqual(self.z_q.integrate(),Quantity(2,Dimension("L")))
        self.assertEqual(2*self.z_q.integrate(),2*Quantity(2,Dimension("L")))

    def test_150_mean(self):
        self.assertEqual(self.z_q.mean(),Quantity(1,Dimension("L")))

    def test_160_sum(self):
        self.assertEqual(self.z_q._sum(),Quantity(3,Dimension("L")))

    def test_170_str(self):
        self.assertEqual(str(Quantity(np.array([1,2,3]),Dimension(None))),
                         "[1 2 3]")

    def test_180_repr(self):
        self.assertEqual(repr(Quantity(1, Dimension("L"))), "<Quantity : 1 m>")
        self.assertEqual(repr(Quantity(np.array([1,2,3]), Dimension("L"))), "<Quantity : [1 2 3] m>")

    def test_190_format(self):
        self.assertEqual("{!s}".format(Quantity(1, Dimension("L"))), "1 m")
        self.assertEqual("{!r}".format(Quantity(1, Dimension("L"))), "<Quantity : 1 m>")

    def test_init_SI_init(self):
        # Not checking symbols
        self.assertEqual(SI_units["kg"],Quantity(1,Dimension("M")))
        self.assertEqual(SI_units["m"],Quantity(1,Dimension("L")))
        self.assertEqual(SI_units["s"],Quantity(1,Dimension("T")))
        self.assertEqual(SI_units["K"],Quantity(1,Dimension("theta")))
        self.assertEqual(SI_units["cd"],Quantity(1,Dimension("J")))
        self.assertEqual(SI_units["A"],Quantity(1,Dimension("I")))
        self.assertEqual(SI_units["mol"],Quantity(1,Dimension("N")))

    def test_eq_ne(self):
        self.assertEqual(self.x_q,self.x_q)
        self.assertTrue(np.all(self.y_q == self.y_q))
        self.assertEqual(self.x_q == self.x_q, True)
        self.assertFalse(self.x_q != self.x_q)
        self.assertTrue(np.all(self.y_q == self.y_q))
        #self.assertFalse(np.all(self.y_q != self.y_q))        
        self.assertFalse(np.all(self.x_q == self.y_q))
        self.assertTrue(np.all(self.x_q != self.y_q))
    
    def test_lt_gt_le_ge(self):
        self.assertTrue(self.x_q <= self.x_q)
        self.assertTrue(np.all(self.y_q <= self.y_q))
        self.assertTrue(np.all(self.x_q >= self.x_q))
        self.assertTrue(np.all(self.y_q >= self.y_q))
        self.assertTrue(np.all(self.x_q < 2*self.x_q))
        self.assertTrue(np.all(self.y_q < 2*self.y_q))
        self.assertTrue(np.all(self.x_q > 0.5*self.x_q))
        self.assertTrue(np.all(self.y_q > 0.5*self.y_q))
        
        
        
    
    def test_units(self):
        Newton = (1 * kg) * (m * s**-2)
        self.assertEqual(units["N"], Newton)
        self.assertEqual(str(units["N"].symbol),"N")
        
    def test_make_quantity(self):
        q = self.x_q
        q.symbol = 'jojo'
        self.assertEqual(q,make_quantity(self.x_q, symbol='jojo'))
        self.assertEqual(str(q),str(make_quantity(self.x_q, symbol='jojo')))
        self.assertEqual(str(q.symbol), 'jojo')
        
        q = self.x_q
        mum = Quantity(0.000001,Dimension("L"), symbol="mum")
        q.favunit = mum
        self.assertEqual(q,make_quantity(self.x_q, favunit=mum))
        self.assertEqual(str(q),str(make_quantity(self.x_q, favunit=mum)))
        self.assertEqual(str(q.symbol), 'UndefinedSymbol')
        
    def test_quad(self):
        
        def func_1(x):
            return 1
        def func_2(x):
            return 1*kg
        
        self.assertEqual(1, quad(func_1, 0, 1)[0])
        self.assertEqual(1*m, quad(func_1, 0*m, 1*m)[0])
        self.assertEqual(1*m*kg, quad(func_2, 0*m, 1*m)[0])

    def test_dblquad(self):
        def func2D(y,x):
            #testing dimensions awareness
            z = y + 1*kg
            zz = x + 1*m
            return 1*kg
        
        self.assertAlmostEqual(4*kg**2*m, dblquad(func2D, 0*m, 2*m, 0*kg, 2*kg)[0])
        
    def test_410_exp_zero(self):
        self.assertEqual(self.x_q ** 0, 1)
        
    #def test_custom_units(self):
    #    from math import pi
    #   self.assertEqual(custom_units['deg'], Quantity(pi /180, Dimension('rad')))
    def test_trigo(self):
        #print(np.cos(self.x_q/self.x_q * 0))
        self.assertEqual(1, np.cos(self.x_q/self.x_q * 0))
        with self.assertRaises(DimensionError):
            np.cos(self.x_q)
    
    def test_400_complex(self):
        self.assertEqual((1j+1) * m, Quantity((1j+1), Dimension("L")))
        self.assertEqual((1j+1) * m + 1 * m, Quantity((1j+2), Dimension("L")))
        self.assertEqual((2j+4) * m + (5j-1) * m, Quantity((7j+3), Dimension("L")))
        
if __name__ == "__main__":
    unittest.main()
        
        