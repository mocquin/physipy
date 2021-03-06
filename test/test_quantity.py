###############################################################################
import numpy as np
import unittest
from fractions import Fraction

import matplotlib
import matplotlib.pyplot


from physipy.quantity import Dimension, Quantity, DimensionError
#from quantity import DISPLAY_DIGITS, EXP_THRESHOLD
from physipy.quantity import vectorize, quad, dblquad, tplquad, brentq #turn_scalar_to_str
from physipy.quantity.calculus import xvectorize, ndvectorize, root
from physipy.quantity import SI_units, units#, custom_units
from physipy.quantity import m, s, kg, A, cd, K, mol
from physipy.quantity import quantify, make_quantity, dimensionify
from physipy.quantity import check_dimension, set_favunit, dimension_and_favunit, drop_dimension, add_back_unit_param, decorate_with_various_unit, array_to_Q_array
from physipy.quantity.utils import asqarray
from physipy import imperial_units, setup_matplotlib
from physipy.quantity.utils import qarange

km = units["km"]
sr = units["sr"]

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
        
    def test_03_test_Quantity_creation(self):
        
        # creation
        # from list
        self.assertTrue(np.all(Quantity([1, 2, 3], Dimension("L")) ==
                               np.array([1, 2, 3])*m))
        self.assertTrue(np.all(Quantity([1], Dimension("L")) ==
                              np.array([1])*m))
        # from tuple
        self.assertTrue(np.all(Quantity((1, 2, 3), Dimension("L")) ==
                               np.array([1, 2, 3])*m))
        self.assertTrue(np.all(Quantity((1,), Dimension("L")) ==
                               np.array([1])*m))
        
        # from ndarray
        self.assertTrue(np.all(Quantity(np.array([1, 2, 3]), Dimension("L")) ==
                               np.array([1, 2, 3])*m))
        self.assertTrue(np.all(Quantity(np.array([1]), Dimension("L")) ==
                               np.array([1])*m))
        self.assertTrue(np.all(Quantity(np.array(1), Dimension("L")) ==
                               np.array(1)*m))
        
        
        
        # equivalence
        self.assertTrue(np.all(Quantity([1, 2, 3], Dimension("L")) ==
                               Quantity((1, 2, 3), Dimension("L"))))
        self.assertTrue(np.all(Quantity([1, 2, 3], Dimension("L")) ==
                                Quantity(np.array([1, 2, 3]), Dimension("L"))))
        
        # one-element array
        left_np = np.array([1])
        right_np = np.array(1)
        left = Quantity(left_np, Dimension("L"))
        right = Quantity(right_np, Dimension("L"))
        self.assertTrue(np.all((left_np == right_np) == (left == right)))
        
        # scalar (not array)
        left_np = 1
        right_np = np.array(1)
        type_np = type(left_np == right_np)
        eq_np = left_np == right_np
        left = Quantity(left_np, Dimension("L"))
        right = Quantity(right_np, Dimension("L"))
        type_q = type(left == right)
        eq_q = left == right
        self.assertEqual(type_np, type_q)
        self.assertEqual(eq_np, eq_q)
        
        
        
        # by multiplicatin value with unit
        self.assertEqual(3*m, 
                        Quantity(3, Dimension("L")))
        self.assertEqual(3.0*m, 
                        Quantity(3.0, Dimension("L")))
        self.assertEqual(Fraction(1, 2)*m,
                        Quantity(Fraction(1, 2), Dimension("L")))
        
        self.assertTrue(np.all([1, 2, 3]*m ==
                              Quantity([1, 2, 3], Dimension("L"))))
               
        self.assertTrue(np.all((1, 2, 3)*m ==
                              Quantity((1, 2, 3), Dimension("L"))))
        
        self.assertTrue(np.all(np.array([1, 2, 3])*m ==
                              Quantity(np.array([1, 2, 3]), Dimension("L"))))
        
        
        
    def test_05_test_units(self):
        self.assertEqual(units["m"],  Quantity(1, Dimension("L")))
        self.assertEqual(units["mm"], Quantity(0.001, Dimension("L")))
        self.assertEqual(units["km"], Quantity(1000, Dimension("L")))
        self.assertEqual(units["Mm"], Quantity(1000000, Dimension("L")))
        
        self.assertEqual(units["kg"],  Quantity(1, Dimension("M")))
        self.assertEqual(units["g"], Quantity(0.001, Dimension("M")))
        
        self.assertEqual(units["K"],  Quantity(1, Dimension("theta")))
        self.assertEqual(units["mK"], Quantity(0.001, Dimension("theta")))
        self.assertEqual(units["kK"], Quantity(1000, Dimension("theta")))
        self.assertEqual(units["MK"], Quantity(1000000, Dimension("theta")))
        
    def test_10_test_produit_division(self):
        self.assertEqual(2 * self.x_q, Quantity(2 * self.x, self.dim))
        self.assertEqual(self.x_q * 2, Quantity(self.x * 2, self.dim))
        self.assertEqual(2 / self.x_q, Quantity(2 / self.x, 1/ self.dim))
        self.assertEqual(self.x_q / 2, Quantity(self.x / 2, self.dim))
        
    def test_15_abs(self):
        left = Quantity(-1, Dimension("L"))
        right = m
        self.assertTrue(abs(left) == right)
        self.assertFalse(abs(left).symbol == right.symbol)
        
        
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
        interp = np.interp
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
        
        
        # set item
        with self.assertRaises(DimensionError):
            u[4] = 35*Quantity(1,Dimension("M"))  
            
        # scalar quantity shouldn't be iterable
        with self.assertRaises(TypeError):
            for i in 5*s: print("i")
            
            
    def test_51_iterator_2d(self):
        # check that iteration works
        # on 2d arrays
        ech_t = np.linspace(1, 10, num=10)*s
        ech_d = np.linspace(1, 20, num=20)*m
     
        T, D = np.meshgrid(ech_t, ech_d)
        res = []
        for i in T:
            res.append(i)
        for q in res:
            self.assertTrue(np.all(q == T[0]))

        
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
    
    def test_inverse(self):
        self.assertTrue(Quantity(1, Dimension("M")).inverse() == Quantity(1, Dimension({"M":-1})))
    
    def test_90_getteur(self):
        # Sans unité favorite
        self.assertEqual(str(self.y_q[2]),"3.0 m")
        # Avec unité favorite
        self.assertEqual(str(self.y_qu[2]),"3000.0 mm")


    def test_120_linspace(self):
        
        m = Quantity(1,Dimension("L"))
        #self.assertTrue(np.all(Quantity(np.linspace(1, 2, num=8), 
        #                                Dimension('L')) == linspace(1*m, 2*m, num=8)))
        #with self.assertRaises(DimensionError):
        #    linspace(1*m, 2)
            
            
        # COMPARE TO np.linspace
        m = Quantity(1,Dimension("L"))
        self.assertTrue(np.all(Quantity(np.linspace(1, 2, num=8), 
                                        Dimension('L')) == np.linspace(1*m, 2*m, num=8)))
        with self.assertRaises(DimensionError):
            np.linspace(1*m, 2)
    
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
        self.assertEqual(self.z_q.sum(),Quantity(3,Dimension("L")))

    def test_170_str(self):
        self.assertEqual(str(Quantity(np.array([1,2,3]),Dimension(None))),
                         "[1 2 3]")

    def test_180_repr(self):
        self.assertEqual(repr(Quantity(1, Dimension("L"))), 
                         "<Quantity : 1 m>")
        self.assertEqual(repr(Quantity(np.array([1,2,3]), Dimension("L"))),
                         "<Quantity : [1 2 3] m>")

    def test_190_format(self):
        self.assertEqual("{!s}".format(Quantity(1, Dimension("L"))), "1 m")
        self.assertEqual("{!r}".format(Quantity(1, Dimension("L"))), "<Quantity : 1 m>")

        
    def test_round(self):
        self.assertEqual(round(1*s), 1*s)
        self.assertEqual(round(1.0*s), 1.0*s)
        self.assertEqual(round(1.1*s), 1*s)
        self.assertEqual(round(1.9*s), 2*s)
        self.assertEqual(round(1.5*s), 2*s)
        
        self.assertEqual(round(1.123456789*s, 0), 1.0*s)
        self.assertEqual(round(1.123456789*s, 2), 1.12*s)
        
        self.assertEqual(round(1.123456789*s, -0), 1.0*s)
        self.assertEqual(round(1.123456789*s, -1), 0*s)
        
        
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
        
        # comparison to non-quantifiables
        self.assertFalse(self.x_q == "a")
        
    def test_none_comparison(self):
        self.assertTrue(m is not None)
        self.assertTrue(m != None)
        self.assertTrue(Quantity(1, Dimension(None)) != None)
        
    #def test_np_all(self):
    #    self.assertTrue(np.all(Quantity([1, 2, 3], Dimension(None))))
    #    self.assertTrue(np.all(Quantity([0, 0, 0], Dimension(None))))
        
        
    
    def test_lt_gt_le_ge(self):
        self.assertTrue(self.x_q <= self.x_q)
        self.assertTrue(np.all(self.y_q <= self.y_q))
        self.assertTrue(np.all(self.x_q >= self.x_q))
        self.assertTrue(np.all(self.y_q >= self.y_q))
        self.assertTrue(np.all(self.x_q < 2*self.x_q))
        self.assertTrue(np.all(self.y_q < 2*self.y_q))
        self.assertTrue(np.all(self.x_q > 0.5*self.x_q))
        self.assertTrue(np.all(self.y_q > 0.5*self.y_q))
        
    def test_min_max(self):
        
        # max uses iterator, not __max__
        # this should raise smthing like TypeError: 'int' object is not iterable because value is int
        #with self.assertRaises(ValueError):
        #    min(self.x_q)
        #with self.assertRaises(ValueError):
        #    max(self.x_q)
        

        # Python's min/max
        self.assertEqual(max(self.y_q), Quantity(max(self.y_q.value), self.y_q.dimension))
        self.assertEqual(min(self.y_q), Quantity(min(self.y_q.value), self.y_q.dimension))
        
        # Numpy's min/max
        self.assertEqual(np.max(self.y_q),
                         Quantity(np.max(self.y_q.value),
                                  self.y_q.dimension))
        self.assertEqual(np.min(self.y_q),
                         Quantity(np.min(self.y_q.value),
                                  self.y_q.dimension))
        self.assertEqual(np.amax(self.y_q),
                         Quantity(np.amax(self.y_q.value),
                                 self.y_q.dimension))
        self.assertEqual(np.amin(self.y_q),
                         Quantity(np.amin(self.y_q.value),
                                  self.y_q.dimension))
        
    def test_has_integer_dimension_power(self):
        self.assertTrue(Quantity(1, Dimension("L")).has_integer_dimension_power())
        self.assertTrue(Quantity(1, Dimension({"L":-2, "M":2})).has_integer_dimension_power())
        self.assertTrue(Quantity(1, Dimension(None)).has_integer_dimension_power())
        
        self.assertFalse(Quantity(1, Dimension({"L":1.2})).has_integer_dimension_power())
    
    def test_units(self):
        Newton = (1 * kg) * (m * s**-2)
        self.assertEqual(units["N"], Newton)
        self.assertEqual(str(units["N"].symbol),"N")
        
    def test_make_quantity(self):
        q = self.x_q.__copy__()
        q.symbol = 'jojo'
        self.assertEqual(q, make_quantity(self.x_q, symbol='jojo'))
        self.assertEqual(str(q), str(make_quantity(self.x_q, symbol='jojo')))
        self.assertEqual(str(q.symbol), 'jojo')
        
        q = self.x_q.__copy__()
        mum = Quantity(0.000001,Dimension("L"), symbol="mum")
        q.favunit = mum
        self.assertEqual(q,make_quantity(self.x_q, favunit=mum))
        self.assertEqual(str(q), str(make_quantity(self.x_q, favunit=mum)))
        self.assertEqual(str(q.symbol), 'UndefinedSymbol')
        
    def test_dimensionify(self):
        self.assertEqual(dimensionify(Dimension("L")), Dimension("L"))
        self.assertEqual(dimensionify(km), Dimension("L"))
        self.assertEqual(dimensionify(1), Dimension(None))
        self.assertEqual(dimensionify(np.array([1, 2, 3])), Dimension(None))
        
    def test_quad(self):
        
        def func_1(x):
            return 1
        def func_2(x):
            return 1*kg
        
        self.assertEqual(1, quad(func_1, 0, 1)[0])
        self.assertEqual(1*m, quad(func_1, 0*m, 1*m)[0])
        self.assertEqual(1*m*kg, quad(func_2, 0*m, 1*m)[0])
        
        
        def toto(x, y):
            return x
        self.assertEqual(0.5,
                        quad(toto, 0, 1, args=(3,))[0])
        self.assertEqual(0.5*m**2,
                        quad(toto, 0*m, 1*m, args=(3,))[0])
        
        def toto(x, y):
            return x*y
        self.assertEqual(1.5,
                        quad(toto, 0, 1, args=(3,))[0])
        self.assertEqual(1.5*m**2*s,
                        quad(toto, 0*m, 1*m, args=(3*s,))[0])
        
    def test_root(self):
        
        def toto(t):
            return -10*s + t
        self.assertEqual(root(toto, 0*s),
                        10*s)
        def tata(t, p):
            return -10*s*p + t
        self.assertEqual(root(tata, 0*s, args=(0.5,)),
                        5*s)
        
    def test_brentq(self):
        def toto(t):
            return -10*s + t
        self.assertEqual(10*s,
                         brentq(toto, -10*s, 10*s))
        
        def tata(t, p):
            return -10*s*p + t
        self.assertEqual(5*s,
                         brentq(tata, -10*s, 10*s, args=(0.5,)))
        
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
    
    def test_300_define_fraction(self):
        self.assertEqual(Fraction(1, 2) * m, Quantity(Fraction(1,2), Dimension("L")))
    
    def test_310_fraction_operation(self):
        self.assertEqual(Fraction(1, 2) * m * 2, Fraction(1, 1) * m)
        self.assertEqual(Fraction(1, 2) * m + Fraction(1, 2) * m, Fraction(1, 1) * m)
        self.assertEqual(Fraction(1, 2) * m / 2, Fraction(1, 4) * m)
        self.assertEqual(2 / (Fraction(1, 2) * m), 4 * 1/m)
        with self.assertRaises(DimensionError):
            Fraction(1, 2) * m + 1
        self.assertTrue(Fraction(1, 2) * m <= Fraction(3, 2) * m)

    def test_400_complex(self):
        self.assertEqual((1j+1) * m, Quantity((1j+1), Dimension("L")))
        self.assertEqual((1j+1) * m + 1 * m, Quantity((1j+2), Dimension("L")))
        self.assertEqual((2j+4) * m + (5j-1) * m, Quantity((7j+3), Dimension("L")))

        
    def test_500_numpy_ufuncs(self):
        
        arr = np.array([1,2,3])
        arr_m = Quantity(arr, Dimension("L"))
        
        # add
        self.assertTrue(np.all(m + arr_m == Quantity(1 + arr, Dimension("L"))))
        self.assertTrue(np.all(arr_m + m == Quantity(1 + arr, Dimension("L"))))
        self.assertTrue(np.all(np.add(arr_m, arr_m) == Quantity(2 * arr, Dimension("L"))))
        
        # sub
        self.assertTrue(np.all(m - arr_m == Quantity(1 - arr, Dimension("L"))))
        self.assertTrue(np.all(arr_m - m == Quantity(arr - 1, Dimension("L"))))
        
        self.assertTrue(np.all(np.subtract(m, arr_m) == Quantity(1 - arr, Dimension("L"))))
        self.assertTrue(np.all(np.subtract(arr_m, m) == Quantity(arr - 1, Dimension("L"))))
        self.assertTrue(np.all(np.subtract(arr_m, arr_m) == Quantity(0 * arr, Dimension("L"))))
        
        # mul
        self.assertTrue(np.all(m * arr_m ==  Quantity(1 * arr, Dimension({"L":2}))))
        self.assertTrue(np.all(arr_m * m ==  Quantity(arr * 1, Dimension({"L":2}))))
        
        self.assertTrue(np.all(np.multiply(m, arr_m) ==  Quantity(1 * arr, Dimension({"L":2}))))
        self.assertTrue(np.all(np.multiply(arr_m, m) == Quantity(arr * 1, Dimension({"L":2}))))
        self.assertTrue(np.all(np.multiply(arr_m, arr_m) == Quantity(arr * arr, Dimension({"L":2}))))
        
        # div
        self.assertTrue(np.all(m / arr_m == np.array([1/1, 1/2, 1/3])))
        self.assertTrue(np.all(arr_m / m == np.array([1.,2.,3.])))
        
        self.assertTrue(np.all(np.divide(m, arr_m) == np.array([1/1, 1/2, 1/3])))
        self.assertTrue(np.all(np.divide(arr_m, m) == np.array([1.,2.,3.])))
        self.assertTrue(np.all(np.divide(arr_m, arr_m) == np.array([1.,1.,1.])))
        
        #logaddexp
        self.assertTrue(np.logaddexp(1, 2) == np.logaddexp(Quantity(1, Dimension(None)), Quantity(2, Dimension(None))))
        
        # true_divide
        self.assertTrue(np.true_divide(3*m, 2*m) == np.true_divide(3, 2))
        self.assertTrue(np.true_divide(3*m, 2*s) == np.true_divide(3*m, 2*s))
        self.assertTrue(np.true_divide(3*m, 2) == np.true_divide(3, 2)*m)

        # floor_divide
        self.assertTrue(np.floor_divide(3*m, 2*m) == np.floor_divide(3, 2))
        with self.assertRaises(DimensionError):
            np.floor_divide(3*m, 2*s)
            
        # negative
        self.assertTrue(np.negative(3*m) == np.negative(3)*m)
        self.assertTrue(np.negative(Quantity(3, Dimension(None))) == np.negative(3)*Quantity(1, Dimension(None)))
        
        # remainder
        self.assertEqual(np.remainder(5*m, 2*m),
                         1*m)
        self.assertTrue(np.all(np.remainder(np.arange(7)*m,
                                      5*m) == np.array([0, 1, 2, 3, 4, 0, 1])*m))
        
        # mod is an alias of remainder
        self.assertEqual(np.mod(5*m, 2*m),
                         1*m)
        self.assertTrue(np.all(np.mod(np.arange(7)*m,
                                      5*m) == np.array([0, 1, 2, 3, 4, 0, 1])*m))
        
        # fmod
        self.assertTrue(np.all(np.fmod([5, 3]*m, [2, 2.]*m)==np.array([1, 1])*m))
        
        # floor
        self.assertTrue(np.floor(3.4*m)==np.floor(3.4)*m)
        
        # ceil
        self.assertTrue(np.ceil(3.4*m)==np.ceil(3.4)*m)
        
        # trunc
        self.assertTrue(np.trunc(3.4*m)==np.trunc(3.4)*m)
        
        
        # absolute
        self.assertEqual(np.absolute(5*m), 5*m)
        self.assertEqual(np.absolute(-5*m), 5*m)
        self.assertTrue(np.all(np.absolute(np.arange(-5, 5)*m)==np.array([5, 4, 3, 2, 1, 0, 1, 2, 3, 4])*m))
        
        # rint
        self.assertEqual(np.rint(5.3*m), 5*m)
        self.assertTrue(np.all(np.rint(np.array([1.1, 2.3, 3.5, 4.6])*m) == np.array([1, 2, 4, 5])*m))
        
        # sign
        self.assertEqual(np.sign(-3*m), -1)
        self.assertEqual(np.sign(3*m), 1)
        
        self.assertTrue(np.all(np.sign(np.array([-1, 0, 1])*m)==np.array([-1, 0, 1])))
        
        # conj and conjugate : conj is an alias for conjugate
        self.assertEqual(np.conj(3*m), 3*m)
        self.assertEqual(np.conj(3*m+2j*m), 3*m-2j*m)
        self.assertEqual(np.conjugate(3*m), 3*m)
        self.assertEqual(np.conjugate(3*m+2j*m), 3*m-2j*m)
        self.assertTrue(np.all(np.conj(np.arange(3)*m+1j*m)==np.arange(3)*m-1j*m))
        self.assertTrue(np.all(np.conjugate(np.arange(3)*m+1j*m)==np.arange(3)*m-1j*m))
        
        # exp
        with self.assertRaises(DimensionError):
            np.exp(3*m)
        with self.assertRaises(DimensionError):
            np.exp(np.arange(3)*m)
            
        self.assertEqual(np.exp(3*m/m), np.exp(3))
        
        # exp2
        with self.assertRaises(DimensionError):
            np.exp2(3*m)
        with self.assertRaises(DimensionError):
            np.exp2(np.arange(3)*m)
        
        
        # log
        with self.assertRaises(DimensionError):
            np.log(3*m)
        with self.assertRaises(DimensionError):
            np.log(np.arange(3)*m)
        
        # log2
        with self.assertRaises(DimensionError):
            np.log2(3*m)
        with self.assertRaises(DimensionError):
            np.log2(np.arange(3)*m)
        
        # log10
        with self.assertRaises(DimensionError):
            np.log10(3*m)
        with self.assertRaises(DimensionError):
            np.log10(np.arange(3)*m)
        
        # expm1
        with self.assertRaises(DimensionError):
            np.expm1(3*m)
        with self.assertRaises(DimensionError):
            np.expm1(np.arange(3)*m)
        
        
        # logp1
        with self.assertRaises(DimensionError):
            np.log1p(3*m)
        with self.assertRaises(DimensionError):
            np.log1p(np.arange(3)*m)
        
        # square
        self.assertEqual(np.square(3*m), (3*m)**2)
        self.assertTrue(np.all(np.square(np.arange(3)*m) ==(np.arange(3)*m)**2))
        
        # cbrt
        self.assertEqual(np.cbrt((3*m)),
                        (3*m)**(1/3))
        self.assertTrue(np.allclose(np.cbrt((np.arange(3)*m)).value, ((np.arange(3)*m)**(1/3)).value))        
        
        # reciprocal
        self.assertEqual(np.reciprocal(3.1*m),
                        1/(3.1*m))
        self.assertTrue(np.all(np.reciprocal(np.array([1.1, 2.3])*m)==
                        1/(np.array([1.1, 2.3])*m)))
        
        
        
        # pow
        with self.assertRaises(TypeError):
            np.power(m, arr_m)
        with self.assertRaises(TypeError):
            np.power(arr_m, m)
        self.assertTrue(np.all(arr_m ** 1 == arr_m))
        self.assertTrue(np.all(arr_m ** 2 == arr_m * arr_m))
        
        # hypot
        self.assertTrue(np.all(np.hypot(m, m) == Quantity((1+1)**(1/2), Dimension("L"))))
        self.assertTrue(np.all(np.hypot(m, arr_m) == Quantity(np.hypot(1, np.array([1,2,3])), Dimension("L"))))
        self.assertTrue(np.all(np.hypot(arr_m, m) == np.hypot(m, arr_m)))
        
        # greater
        self.assertTrue(np.all(np.greater(m, m) == False))
        self.assertTrue(np.all(np.greater(m, arr_m) == np.array([False, False, False])))
        self.assertTrue(np.all(np.greater(arr_m, m) == np.array([False, True, True])))
        self.assertTrue(np.all(np.greater(arr_m, arr_m) == np.array([False, False, False])))
        
        # greater_or_equal
        self.assertTrue(np.all(np.greater_equal(m, m) == True))
        self.assertTrue(np.all(np.greater_equal(m, arr_m) == np.array([True, False, False])))
        self.assertTrue(np.all(np.greater_equal(arr_m, m) == np.array([True, True, True])))
        self.assertTrue(np.all(np.greater_equal(arr_m, arr_m) == np.array([True, True, True])))
        
        # less
        self.assertTrue(np.all(np.less(m, m) == False))
        self.assertTrue(np.all(np.less(m, arr_m) == np.array([False, True, True])))
        self.assertTrue(np.all(np.less(arr_m, m) == np.array([False, False, False])))
        self.assertTrue(np.all(np.less(arr_m, arr_m) == np.array([False, False, False])))
        
        # less_or_equal
        self.assertTrue(np.all(np.less_equal(m, m) == True))
        self.assertTrue(np.all(np.less_equal(m, arr_m) == np.array([True, True, True])))
        self.assertTrue(np.all(np.less_equal(arr_m, m) == np.array([True, False, False])))
        self.assertTrue(np.all(np.less_equal(arr_m, arr_m) == np.array([True, True, True])))
        
        # equal
        self.assertTrue(np.all(np.equal(arr_m, arr_m)))
        self.assertTrue(np.all(np.equal(m, m)))
        
        # not_equal
        self.assertTrue(np.all(np.not_equal(arr_m, 0*m)))
        self.assertTrue(np.all(np.not_equal(m, 0*m)))
        
        # isreal
        #self.assertTrue(np.isreal(m))
        #self.assertTrue(np.all(np.isreal(arr_m)))
        
        # isfinite
        self.assertTrue(np.isfinite(m))
        self.assertTrue(np.all(np.isfinite(arr_m)))
        
        # isinf
        self.assertFalse(np.isinf(m))
        self.assertFalse(np.all(np.isinf(arr_m)))
        
        # isnan
        self.assertFalse(np.isnan(m))
        self.assertFalse(np.all(np.isnan(arr_m)))
        
        # copysign
        self.assertEqual(np.copysign(m, 1), m)
        self.assertEqual(np.copysign(m, -1), -m)
        
        self.assertTrue(np.all(np.copysign(m, arr_m) == np.array([1,1,1])*m))
        self.assertTrue(np.all(np.copysign(m, -arr_m) == -np.array([1,1,1])*m))
        
        # nextafter
        eps = np.finfo(np.float64).eps
        self.assertTrue(np.nextafter(1*m, 2*m)== eps*m+1*m)
        
        # modf
        res_modf = np.modf([0, 3.5]*m)
        frac, integ = res_modf
        self.assertTrue(np.all(frac == np.array([0, 0.5])*m))
        self.assertTrue(np.all(integ == np.array([0, 3])*m))
        
        
        # sqrt
        self.assertEqual(np.sqrt(m), Quantity(1, Dimension({"L":1/2})))
        self.assertTrue(np.all(np.sqrt(arr_m) == Quantity(np.sqrt(np.array([1.,2.,3.])), Dimension({"L":1/2}))))

        ## Trigo
        zero_rad = Quantity(0, Dimension("RAD"))
        zero_none = Quantity(0, Dimension(None))
        one_none  = Quantity(1, Dimension(None))
        # cos
        self.assertTrue(np.cos(zero_none) == np.cos(zero_rad))
        self.assertTrue(np.cos(zero_none) == np.cos(0))
        # cosh
        self.assertTrue(np.cosh(zero_none) == np.cosh(zero_rad))
        self.assertTrue(np.cosh(zero_none) == np.cosh(0))
        # sin
        self.assertTrue(np.sin(zero_none) == np.sin(zero_rad))
        self.assertTrue(np.sin(zero_none) == np.sin(0))
        # sinh
        self.assertTrue(np.sinh(zero_none) == np.sinh(zero_rad))
        self.assertTrue(np.sinh(zero_none) == np.sinh(0))
        # tan
        self.assertTrue(np.tan(zero_none) == np.tan(zero_rad))
        self.assertTrue(np.tan(zero_none) == np.tan(0))
        # tanh
        self.assertTrue(np.tanh(zero_none) == np.tanh(zero_rad))
        self.assertTrue(np.tanh(zero_none) == np.tanh(0))
        # arccos
        self.assertTrue(np.arccos(zero_none) == np.arccos(0))
        # arccosh
        self.assertTrue(np.arccosh(one_none) == np.arccosh(1))
        # arcsin
        self.assertTrue(np.arcsin(zero_none) == np.arcsin(0))
        # arcsinh
        self.assertTrue(np.arcsinh(zero_none) == np.arcsinh(0))
        # arctan
        self.assertTrue(np.arctan(zero_none) == np.arctan(0))
        # arctanh
        self.assertTrue(np.arctanh(zero_none) == np.arctanh(0))
        # arctan2
        self.assertTrue(np.arctan2(one_none, one_none) == np.arctan2(1, 1))
        
        # fabs
        self.assertEqual(np.fabs(m), m)
        self.assertEqual(np.fabs(-m), m)
        self.assertTrue(np.all(np.fabs(arr_m) == arr_m))
        self.assertTrue(np.all(np.fabs(-arr_m) == arr_m))
        self.assertTrue(np.all(np.fabs(Quantity(np.array([-1, 0, 1]), Dimension("L"))) == Quantity(np.array([1, 0, 1]), Dimension("L"))))
        
    def test_510_numpy_functions(self):
        
        arr = np.array([1,2,3])
        arr_m = Quantity(arr, Dimension("L"))
        
        # np.alen
        self.assertEqual(np.alen(3*m), np.alen(3))
        self.assertEqual(np.alen(np.arange(3)*m), np.alen(np.arange(3)))
        
        # np.all
        
        # np.allclose
        
        # np.amax
        self.assertEqual(np.amax(self.y_q),
                         Quantity(np.amax(self.y_q.value),
                                 self.y_q.dimension))
        
        # np.amin
        self.assertEqual(np.amin(self.y_q),
                         Quantity(np.amin(self.y_q.value),
                                  self.y_q.dimension))
        
        # np.any
        self.assertTrue(np.all(np.append(3*m, 4*m) == np.array([3, 4])*m))
        with self.assertRaises(DimensionError):
            np.append(3*m, 4*kg)
        self.assertTrue(np.all(np.append(Quantity(1, Dimension(None)), 1) == Quantity([1, 1], Dimension(None))) )
        
        # np.argmax
        self.assertEqual(np.argmax(np.arange(5)*m), 4*m)
        
        # np.argmin
        self.assertEqual(np.argmin(np.arange(5)*m), 0*m)
        
        # np.argsort
        self.assertTrue(np.all(np.argsort(np.arange(5)*m)==np.arange(5)))

        # np.around
        self.assertTrue(np.all(np.around(np.linspace(2.2, 3.656, 10)*m) == np.around(np.linspace(2.2, 3.656, 10))*m))
        
        # np.atleast_1d
        self.assertTrue(np.all(np.atleast_1d(4*m) == np.array([4])*m))
        self.assertTrue(np.all(np.atleast_1d(np.arange(3)*m) == np.arange(3)*m))
        left_in = np.arange(5)*m
        right_in = 4*m
        res_in1, res_in2 = np.atleast_1d(left_in, right_in)
        self.assertTrue(np.all(res_in1 == np.arange(5)*m))
        self.assertTrue(np.all(res_in2 == np.array([4])*m))
        
        # np.average
        self.assertEqual(np.average(arr_m), 2*m)
        self.assertEqual(np.average(5*m), 5*m)
        
        # np.broadcast_arrays
        x = np.array([[1,2,3]])*m
        y = np.array([[4],[5]])*K
        res = np.broadcast_arrays(x, y)
        self.assertTrue(
            np.all(res[0] == np.array([[1, 2, 3],
                                       [1, 2, 3]])*m
                  )
        )
        self.assertTrue(
            np.all(res[1] == np.array([[4, 4, 4],
                                       [5, 5, 5]])*K
                  )
        )
        
        x = np.array([[1,2,3]])*m
        y = np.array([[4],[5]])
        res = np.broadcast_arrays(x, y)

        self.assertTrue(
            np.all(res[0] == np.array([[1, 2, 3],
                                         [1, 2, 3]])*m
                  )
        )
        self.assertTrue(
            np.all(res[1] == np.array([[4, 4, 4],
                                       [5, 5, 5]])
                  )
        )
        
        # np.block
        #A = np.eye(2) * 2 * m
        #B = np.eye(3) * 3
        #res = np.block([
        #    [A,               np.zeros((2, 3))*m],
        #    [np.ones((3, 2))*m, B*m               ]
        #])
        #exp = array(
        #    [[2., 0., 0., 0., 0.],
        #    [0., 2., 0., 0., 0.],
        #    [1., 1., 3., 0., 0.],
        #    [1., 1., 0., 3., 0.],
        #    [1., 1., 0., 0., 3.]
        #    ])*m
        #self.assertTrue(np.all(res, exp))
        
        # np.clip
        self.assertTrue(np.all(np.clip(np.arange(10)*m,
                                       2*m, 
                                       7*m) == np.array([2, 2, 2, 3, 4, 5, 6, 7, 7, 7])*m))
        with self.assertRaises(DimensionError):
            np.clip(np.arange(10)*m, 2, 3)
        with self.assertRaises(DimensionError):
            np.clip(np.arange(10)*m, 2*m, 3)
        with self.assertRaises(DimensionError):
            np.clip(np.arange(10)*m, 2, 3*kg)
        
        
        # np.column_stack
        a = np.array([1, 2, 3])*m
        b = np.array([2, 3, 4])*m
        self.assertTrue(np.all(np.column_stack((a, b))==np.array([[1, 2],
        [2, 3],
        [3, 4]])*m))
        
        # np.compress
        a = np.array([[1, 2], [3, 4], [5, 6]])*m
        res = np.compress([0, 1], a, axis=0)
        exp = np.array([[3, 4]])*m
        self.assertTrue(np.all(res == exp))
        
        # np.concatenate
        a = np.array([[1, 2], [3, 4]])*m
        b = np.array([[5, 6]])*m
        self.assertTrue(np.all(np.concatenate((a, b), axis=0)==np.array([[1, 2],
       [3, 4],
       [5, 6]])*m))
        
        # np.copy
        self.assertTrue(np.all(np.copy(np.arange(3)*m) == np.arange(3)*m))
        
        # np.cross
        self.assertTrue(np.all(np.cross(np.array([[1, 2, 3]])*m,
                                       np.array([[4, 5, 6]])*m)== np.array([-3, 6, -3])*m**2))
        

        # np.cumprod
        
        # np.cumsum
        self.assertTrue(np.all(np.cumsum(np.arange(3)*m)==np.array([0, 1, 3])*m))
        
        # np.diagonal
        a = np.arange(4).reshape(2,2)*m
        self.assertTrue(np.all(np.diagonal(a)== np.array([0,3])*m))
        
        # np.diff
        # a = np.array([1, 2, 4, 7, 0])*m
        # self.assertTrue(np.all(np.diff(a)==np.array([1, 2, 3, -7])*m))
        
        
        # np.dot
        a = np.array([[1, 0], [0, 1]])*m
        b = np.array([[4, 1], [2, 2]])*m
        self.assertTrue(np.all(np.dot(a, b)==np.array([[4, 1],
       [2, 2]])*m**2))
        
        # np.dstack
        a = np.array((1,2,3))*m
        b = np.array((2,3,4))*m
        self.assertTrue(np.all(np.dstack((a, b)) == np.array([[1, 2],
        [2, 3],
        [3, 4]])*m))
        
        # np.ediff1d
        
        # np.sum
        self.assertEqual(np.sum(arr_m), 6 * m)
        self.assertEqual(np.sum(5*m), 5 * m)
        self.assertTrue(np.all(np.sum(np.array([[1, 2, 3],
                                         [1, 2, 3]])*m, axis=1) ==  np.array([6, 6])*m))
        
        # np.mean
        self.assertEqual(np.mean(arr_m), 2*m)
        self.assertEqual(np.mean(5*m), 5*m)
        self.assertTrue(np.all(np.mean(np.arange(6).reshape(3,2)*s, axis=1) == np.mean(np.arange(6).reshape(3,2), axis=1)*s))
        
        # np.std
        self.assertEqual(np.std(arr_m), 0.816496580927726*m)
        self.assertEqual(np.std(5*m), 0*m)

        
        # np.median
        self.assertEqual(np.median(arr_m), 2*m)
        self.assertEqual(np.median(5*m), 5*m)
        
        # np.var
        self.assertEqual(np.var(arr_m), 0.6666666666666666*m**2)
        self.assertEqual(np.var(5*m), 0*m**2)
        
        # np.trapz
        self.assertEqual(np.trapz(arr_m), 4*m)
        self.assertEqual(np.trapz(arr_m, dx=1*m), 4*m**2)
        
        
        # np.linspace
        self.assertTrue(np.all(np.linspace(0*m, 5*m) == Quantity(np.linspace(0, 5), Dimension("L"))))
        
        # np.max
        self.assertEqual(np.max(self.y_q),
                         Quantity(np.max(self.y_q.value),
                                  self.y_q.dimension))
        
        # np.min
        self.assertEqual(np.min(self.y_q),
                         Quantity(np.min(self.y_q.value),
                                  self.y_q.dimension))
        
        # np.interp
        self.assertEqual(np.interp(1*m, arr_m, arr_m**2), 
                         1*m**2)
        self.assertEqual(np.interp(1.5*m, arr_m, arr_m**2), 
                         (2.5)*m**2)
        self.assertEqual(np.interp(1.5*m, arr_m, arr_m**2, left=0*m**2), 
                         2.5*m**2)
        
        with self.assertRaises(DimensionError):
            np.interp(1*m, arr_m, arr_m**2, left=0*m)
        
        with self.assertRaises(DimensionError):
            np.interp(1*s, arr_m, arr_m**2)
            
        
    def test_real(self):
        self.assertEqual(m.real, 1*m)
        a = np.array([1, 2, 3])*m
        self.assertTrue(np.all(a.real == np.array([1, 2, 3])*m))
        
        
    def test_imag(self):
        self.assertEqual(m.imag, 0*m)
        a = np.array([1, 2, 3])*m
        self.assertTrue(np.all(a.imag == np.array([0, 0, 0])*m))
        
    def test_transpose(self):
        a = np.array([[1, 0], [0, 1]])*m
        self.assertTrue(np.all(a.T == np.array([[1, 0], [0, 1]]).T*m))
        
        
        with self.assertRaises(AttributeError):
            # a = 5
            # a.T raises AttributeError
            q = 5*m
            q.T

    

    def test_sum_builtin(self):
        # on list of 2 scalar-value-quantity
        self.assertEqual(sum([self.x_q, self.x_q], 0*self.x_q), 2*self.x_q)
        with self.assertRaises(DimensionError):
            sum([self.x_q, self.x_q])
        # on array-value-quantity
        self.assertEqual(sum(self.y_q, 0*m), sum(self.y)*m)
        with self.assertRaises(DimensionError):
            sum(self.y_q)
        # mixin
        self.assertTrue(np.all(sum([self.x_q, self.y_q], 0*m) == sum([self.x, self.y])*m))
        with self.assertRaises(DimensionError):
            sum([self.x_q, self.y_q])

    def test_500_decorator_check_dimension(self):
        
        # To check the dimension analysis of inputs
        # Two inputs, one output
        def speed(l, t):
            return l/t
        wrapped_speed = check_dimension((m, s), m/s)(speed)
        
        with self.assertRaises(DimensionError):
            wrapped_speed(1, 1) # a scalar is interpreted as dimensionless
        with self.assertRaises(DimensionError):
            wrapped_speed(1*m, 1)
        with self.assertRaises(DimensionError):
            wrapped_speed(1, 1*s)
            
        # To check that the decorator does not alterate the returned value
        self.assertEqual(wrapped_speed(1*m, 1*s), 1*m / (1*s))
        
        # To check the dimension analysis of outputs
        def wrong_speed(l, t):
            return l*t
        wrappred_wrong_speed = check_dimension((m, s), m/s)(wrong_speed)    
        with self.assertRaises(DimensionError):
            wrappred_wrong_speed(m, s)
        
        
        # with Dimension notation
        def speed(l, t):
            return l/t
        wrapped_speed = check_dimension(("L", "T"), 'L/T')(speed)

        with self.assertRaises(DimensionError):
            wrapped_speed(1, 1) # a scalar is interpreted as dimensionless
        with self.assertRaises(DimensionError):
            wrapped_speed(1*m, 1)
        with self.assertRaises(DimensionError):
            wrapped_speed(1, 1*s)
        self.assertEqual(wrapped_speed(1*m, 1*s), 1*m / (1*s))
        
    def test_501_decorator_favunit(self):
        def speed(l, t):
            return l/t
        mph = imperial_units["mil"] / units["h"]
        mph.symbol = "mph"
        mph_speed = set_favunit(mph)(speed)
        
        # check that the actual value is the same
        self.assertEqual(speed(m, s), mph_speed(m, s))
        
        # check the favunits
        self.assertEqual((mph_speed(m, s)).favunit, mph)
        
    def test_502_decorator_dimension_and_favunit(self):
        def speed(l, t):
            return l/t
        mph = imperial_units["mil"] / units["h"]
        mph.symbol = "mph"
        
        # value
        self.assertEqual(dimension_and_favunit((km, s), mph)(speed)(5*m, 2*s), speed(5*m, 2*s))
        
        # favunit
        self.assertEqual(dimension_and_favunit((km, s), mph)(speed)(5*m, 2*s).favunit, mph)
        
        # dimension check
        with self.assertRaises(DimensionError):
            dimension_and_favunit((km, s), mph)(speed)(5*s, 2*s)
            
    def test_503_decorator_drop_dimension(self):
        # this function will always compare inputs to ints
        # so the inputs must be scalar of dimless Quantitys
        def speed_dimless(l, t):
            if not t==0 and not l<0:
                return l/t
            else:
                return 0

        # check that it indeed fails
        with self.assertRaises(DimensionError):
            speed_dimless(m, s)
            
        # if dimensions are dropped, should work
        # the output is scalar
        self.assertEqual(drop_dimension(speed_dimless)(1*m, 1*s), 1)
        self.assertEqual(drop_dimension(speed_dimless)(5*m, 1*s), 5)
            
    def test_504_decorator_add_back_unit_param(self):
        def speed_dimless(l, t):
            if not t==0 and not l<0:
                return l/t
            else:
                return 0
            
        # inputs must be dimless, but we want to add dimension to output
        with self.assertRaises(DimensionError):
            speed_dimless(5, 1) > 5*m/s
            
        # will multiplu each raw output by m/s
        self.assertEqual(add_back_unit_param(m/s)(speed_dimless)(5, 1), 5*m/s)
    
    def test_505_decorator_decorate_with_various_unit(self):
        
        #interp = decorate_with_various_unit(("A", "A", "B"), ("B"))(np.interp)
        def func(x, y):
            return x+y

        d_func = decorate_with_various_unit(("A", "A"), "A")(func)
        
        # when ok
        self.assertEqual(d_func(1*m, 1*m), 
                        2*m)
        
        # when incoherent inputs
        with self.assertRaises(DimensionError):
            d_func(1*m, 1*s)
            
        def func2(x, y): return x*y
        # this will set the output unit to "A"
        d_func2 = decorate_with_various_unit(("A", "A"), "A")(func2)
        self.assertEqual(d_func2(1*m, 1*m), 
                        1*m)
        
    def test_600_asqarray(self):


        self.assertTrue(np.all(
            asqarray([1*s, 2*s]) == Quantity([1, 2], Dimension("T"))
        ))
        
        self.assertTrue(np.all(
            (asqarray([1., 2.]) == Quantity([1, 2], Dimension(None)))
        ))
        
        self.assertTrue(np.all(
            asqarray( np.array([1*s, 2*s] , dtype=object)) == Quantity([1, 2], Dimension("T"))
        ))
        
        self.assertTrue(np.all(
            asqarray(np.array([1*m], dtype=object)) == Quantity([1], Dimension("L"))
        ))
        
        self.assertTrue(np.all(
            asqarray(np.array([1., 2.])) == Quantity([1, 2], Dimension(None))
        ))
        

        arrq_1 = np.array([1, 2, 3]) * m
        out = asqarray(arrq_1)
        exp = Quantity(np.array([1, 2, 3]), Dimension("L"))
        self.assertTrue(np.all(out == exp))
        
        arrq_2 = np.array([1*m, 2*m, 3*m], dtype=object)
        out = asqarray(arrq_2)
        exp = Quantity(np.array([1, 2, 3]), Dimension("L"))
        self.assertTrue(np.all(out == exp))

        
        arrq_3 = [1*m, 2*m, 3*m]
        out = asqarray(arrq_3)
        exp = Quantity(np.array([1, 2, 3]), Dimension("L"))
        self.assertTrue(np.all(out == exp))

        
        arrq_4 = (1*m, 2*m, 3*m)
        out = asqarray(arrq_4)
        exp = Quantity(np.array([1, 2, 3]), Dimension("L"))
        self.assertTrue(np.all(out == exp))
        
        
        arrq_5 = np.array(1) * m
        out = asqarray(arrq_5)
        exp = Quantity(1, Dimension("L"))
        self.assertTrue(np.all(out == exp))
            
        arrq_7 = [1*m]
        out = asqarray(arrq_7)
        exp = Quantity([1], Dimension("L"))
        self.assertTrue(np.all(out == exp))
            
        arrq_8 = (1*m, )
        out = asqarray(arrq_8)
        exp = Quantity([1], Dimension("L"))
        self.assertTrue(np.all(out == exp))
        
        arrq_9 = np.array([m], dtype=object)
        out = asqarray(arrq_9)
        exp = Quantity(np.array([1]), Dimension("L"))
        self.assertTrue(np.all(out == exp))
        
        arrq_10 = np.array([1, 2, 3])
        out = asqarray(arrq_10)
        exp = Quantity(np.array([1, 2, 3]), Dimension(None))
        self.assertTrue(np.all(out == exp))


        arrq_1 = np.array([1, 2, 3]) 
        out = asqarray(arrq_1)
        exp = Quantity(np.array([1, 2, 3]), Dimension(None))
        self.assertTrue(np.all(out == exp))
        
        arrq_2 = np.array([1, 2, 3], dtype=object)
        out = asqarray(arrq_2)
        exp = Quantity(np.array([1, 2, 3]), Dimension(None))
        self.assertTrue(np.all(out == exp))
        
        
        arrq_3 = [1, 2, 3]
        out = asqarray(arrq_3)
        exp = Quantity(np.array([1, 2, 3]), Dimension(None))
        self.assertTrue(np.all(out == exp))
        
        
        arrq_4 = (1, 2, 3)
        out = asqarray(arrq_4)
        exp = Quantity(np.array([1, 2, 3]), Dimension(None))
        self.assertTrue(np.all(out == exp))
        
        
        arrq_5 = np.array(1)
        out = asqarray(arrq_5)
        exp = Quantity(1, Dimension(None))
        self.assertTrue(np.all(out == exp))
            
        arrq_7 = [1]
        out = asqarray(arrq_7)
        exp = Quantity([1], Dimension(None))
        self.assertTrue(np.all(out == exp))
            
        arrq_8 = (1, )
        out = asqarray(arrq_8)
        exp = Quantity([1], Dimension(None))
        self.assertTrue(np.all(out == exp))
        
        arrq_9 = np.array([1], dtype=object)
        out = asqarray(arrq_9)
        exp = Quantity(np.array([1]), Dimension(None))
        self.assertTrue(np.all(out == exp))
        
        arrq_10 = np.array([1, 2, 3])
        out = asqarray(arrq_10)
        exp = Quantity(np.array([1, 2, 3]), Dimension(None))
        self.assertTrue(np.all(out == exp))

    
    
    
        
    def test_std(cls):
        cls.assertEqual(m.std(), 0.0 * m)
        cls.assertEqual(cls.y_q.std(), Quantity(cls.y.std(), Dimension("L")))
        cls.assertEqual(cls.z_q.std(), Quantity(cls.z.std(), Dimension("L")))
        
    def test_math_sqrt(cls):
        with cls.assertRaises(DimensionError):
            import math
            math.sqrt(m)
            
    def test_math_cos(cls):
        import math
        rad = units["rad"]
        cls.assertEqual(math.cos(rad), math.cos(1))
        
        with cls.assertRaises(DimensionError):
            math.cos(m)
            
    def test_check_dim(self):
        self.assertTrue(m.check_dim(Dimension("L")))
        self.assertFalse(m.check_dim(Dimension("RAD")))
        

    def test_matplotlib(self):

        arr_m = np.linspace(1, 3, 2)*1000 * m
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(np.linspace(1, 3, 2), arr_m, "o")
    
    
    def test_matplotlib_units(self):
        
        setup_matplotlib()
        arr_m = np.linspace(1, 3, 2)*1000 * m
        
        # simple plot without favunit
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        # axis unit should be SI unit : m
        self.assertEqual(ax.yaxis.units, 
                        m)
        self.assertEqual(ax.yaxis.label.get_text(), 
                         "m")
        
        # plot with same dimension favunit (value 1)
        arr_m.favunit = km
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        # axis unit should be favunit : km
        self.assertEqual(ax.yaxis.units, 
                        km)
        self.assertEqual(ax.yaxis.label.get_text(), 
                         "km")
        
        
        # plot with non-unitary, same dimension
        twokm = 2*km
        twokm.symbol = "2km"
        arr_m.favunit = twokm
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        self.assertEqual(ax.yaxis.units, 
                        twokm)
        self.assertEqual(ax.yaxis.label.get_text(), 
                         "2km")
        
        # plot with unitary, different dimension
        #km_per_sr = km/sr
        #km_per_sr.symbol = "km/sr"
        #arr_m.favunit = km_per_sr
        #fig, ax = matplotlib.pyplot.subplots()
        #ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        #self.assertEqual(ax.yaxis.units, 
        #                km_per_sr)
        #self.assertEqual(ax.yaxis.label.get_text(), 
        #                 "km/sr")
        
        
        # plot with non unitary, different dimension
        #two_km_per_sr = 2*km/sr
        #two_km_per_sr.symbol = "2km/sr"
        #arr_m.favunit = two_km_per_sr
        #fig, ax = matplotlib.pyplot.subplots()
        #ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        #self.assertEqual(ax.yaxis.units, 
        #                two_km_per_sr)
        #self.assertEqual(ax.yaxis.label.get_text(), 
        #                 "2km/sr")
        
        
        # trying to plot something with different dimension
        # m, then m**2
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        with self.assertRaises(DimensionError):
            ax.plot(np.linspace(1, 3, 2), arr_m**2, "o")
        
        
        # Dim-less then m**2
        arr_m = np.linspace(1, 3, 2)*1000 * m
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(np.linspace(1, 3, 2), Quantity(np.linspace(1, 3, 2), Dimension(None), "o"))
        with self.assertRaises(DimensionError):
            ax.plot(np.linspace(1, 3, 2), arr_m**2, "o")
        
    
    def test_flat(self):

        # indexing
        arr_m = np.arange(5)*m
        self.assertEqual(arr_m.flat[0],
                        0*m)
        self.assertEqual(arr_m.flat[-1],
                        4*m)
        
        arr_m = np.arange(6).reshape(3, 2)*m
        self.assertEqual(arr_m.flat[0],
                        0*m)
        self.assertEqual(arr_m.flat[-1],
                        5*m)
        
        # iteration
        res = [x for x in np.arange(5)*m]
        for fx, x in zip(arr_m.flat, res):
            self.assertEqual(fx, x)
            
        res = [x for x in np.arange(6)*m]
        for fx, x in zip((np.arange(6).reshape(3, 2)*m).flat, res):
            self.assertEqual(fx, x)
            
    def test_flatten(self):
        q = 5*m
        arr = np.arange(10).reshape(5,2)
        qarr = arr*m
        
        self.assertTrue(np.all(
        qarr.flatten() == np.arange(10)*m
        ))
        
        with self.assertRaises(AttributeError):
            q.flatten()
        
    def test_ravel(self):
        
        exp = np.array([1, 2, 3, 4, 5, 6])*m
        res = np.ravel(np.array([[1, 2, 3], [4, 5, 6]])*m)
        self.assertTrue(np.all(exp == res))
        
        
    def test_reshape(self):
        exp = np.array([1, 2, 3, 4, 5, 6])*m
        res = np.reshape(np.array([[1, 2, 3], [4, 5, 6]])*m, (1, 6))
        self.assertTrue(np.all(exp == res))     
        
    def test_xvectorize(self):
        arr_m = np.arange(5)*m
        
        def thresh(x):
            if x >3*m:
                return x
            else:
                return 3*m
        vec_thresh = xvectorize(thresh)
        
        res = vec_thresh(arr_m)
        exp = np.array([3, 3, 3, 3, 4])*m
        self.assertTrue(np.all(res == exp))
        
    #def test_vectorize(self):
    #    
    #    # 1D array
    #    arr_m = np.arange(5)*m
#
    #    def thresh(x):
    #        if x >3*m:
    #            return x
    #        else:
    #            return 3*m
    #    vec_thresh = vectorize(thresh)
    #    
    #    res = vec_thresh(arr_m)
    #    exp = np.array([3, 3, 3, 3, 4])*m
    #    self.assertTrue(np.all(res == exp))
    #    
    #    # nD array
    #    arr_m = np.arange(6).reshape(3,2)*m
    #    
    #    def thresh(x):
    #        if x >3*m:
    #            return x
    #        else:
    #            return 3*m
    #    vec_thresh = vectorize(thresh)
    #    
    #    res = vec_thresh(arr_m)
    #    exp = np.array([[3, 3],[3, 3], [4, 5]])*m
    #    self.assertTrue(np.all(res == exp))        
        
        
    def test_ndvectorize(self):
        arr_m = np.arange(6).reshape(3,2)*m
        
        def thresh(x):
            if x >3*m:
                return x
            else:
                return 3*m
        vec_thresh = ndvectorize(thresh)
        
        res = vec_thresh(arr_m)
        exp = np.array([[3, 3],[3, 3], [4, 5]])*m
        self.assertTrue(np.all(res == exp))


    def test_np_fft_fftshift(self):
        exp = np.fft.fftshift(np.arange(10))*s
        self.assertTrue(np.all(np.fft.fftshift(np.arange(10)*s)==exp))


    def test_np_fft_ifftshift(self):
        exp = np.fft.ifftshift(np.arange(10))*s
        self.assertTrue(np.all(np.fft.ifftshift(np.arange(10)*s)==exp))
    
        
    def test_np_fft_fft(self):
        exp = np.fft.fft(np.arange(10))*s
        self.assertTrue(np.all(np.fft.fft(np.arange(10)*s)==exp))
        
        
    def test_np_fft_ifft(self):
        exp = np.fft.ifft(np.arange(10))*s
        self.assertTrue(np.all(np.fft.ifft(np.arange(10)*s)==exp))
        
        
    def test_qarange(self):
        self.assertTrue(np.all(qarange(0*s, 1*s, 0.1*s) == np.arange(0, 1, 0.1)*s))
        self.assertTrue(np.all(qarange(0*s, 0.1*s)      == np.arange(0, 0.1)*s))        
        self.assertTrue(np.all(qarange(0*s, step=0.1*s) == np.arange(0, step=0.1)*s))
        self.assertTrue(np.all(qarange(0*s, stop=0.1*s) == np.arange(0, stop=0.1)*s))
        self.assertTrue(np.all(qarange(0*s, step=0.1*s,
                                       stop=2*s)        == np.arange(0, step=0.1, stop=2)*s))

    def test_np_convolve(self):
        arr_s = np.ones(10)*s
        arr = np.ones(10)
        
        self.assertTrue(np.all(np.convolve(arr_s, arr) == np.convolve(np.ones(10), np.ones(10))*s))
        self.assertTrue(np.all(np.convolve(arr_s, arr_s) == np.convolve(np.ones(10), np.ones(10))*s**2))
        
        
if __name__ == "__main__":
    unittest.main()
        
        