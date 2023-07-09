###############################################################################
import numpy as np
import unittest
from fractions import Fraction
import math
import time

import pandas as pd
import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt

from physipy.quantity import Dimension, Quantity, DimensionError
#from quantity import DISPLAY_DIGITS, EXP_THRESHOLD
# from physipy.quantity import vectorize #turn_scalar_to_str
from physipy.calculus import xvectorize, ndvectorize,  quad, dblquad, tplquad, solve_ivp, root, brentq
from physipy.quantity import units, imperial_units  # , custom_units
from physipy.quantity import m, s, kg, A, cd, K, mol
from physipy.quantity import quantify, make_quantity, dimensionify
from physipy.quantity import check_dimension, set_favunit, dimension_and_favunit, drop_dimension, add_back_unit_param, decorate_with_various_unit
from physipy.quantity.utils import asqarray, hard_equal, very_hard_equal, qarange
import physipy

import doctest
from physipy import quantity, constants, math
from physipy import calculus, utils, setup_matplotlib, plotting_context

# The load_tests() function is automatically called by unittest
# see https://docs.python.org/3/library/doctest.html#unittest-api
def load_tests(loader, tests, ignore):
    # /!\ dimension doctest is tested in test_dimension
    tests.addTests(doctest.DocTestSuite(quantity))
    tests.addTests(doctest.DocTestSuite(calculus))
    # TODO : dict and module share the same name
    # tests.addTests(doctest.DocTestSuite(physipy.quantity.units))
    tests.addTests(doctest.DocTestSuite(utils))
    # TODO : dict and module share the same name
    # tests.addTests(doctest.DocTestSuite(constants))
    tests.addTests(doctest.DocTestSuite(math))
    return tests


km = units["km"]
m = units["m"]
sr = units["sr"]
mm = units["mm"]
V = units["V"]


class TestQuantity(unittest.TestCase):

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

        cls.x = 12.1111
        cls.y = np.array([1, 2, 3, 4.56])
        cls.z = np.array([1, 1, 1])
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

        cls.times = []
        cls.ids = []

    @classmethod
    def tearDownClass(cls):
        cls.df = pd.DataFrame.from_dict({
            "time": cls.times,
            "id": cls.ids,
        })

    def test_pickle(self):
        import pickle

        q = 2.345*K / s
        q.symbol = "toto"
        s2 = s**2
        s2.symbol = "s**2"
        q.favunit = s2
        saved_object = pickle.dumps(q)
        new = pickle.loads(saved_object)
        self.assertTrue(very_hard_equal(q, new))

        q = 2.345*K
        q.favunit = s
        saved_object = pickle.dumps(q)
        new = pickle.loads(saved_object)
        self.assertTrue(very_hard_equal(q, new))

    def test_hard_equal(self):
        q1 = Quantity(1, Dimension('L'))
        q2 = Quantity(1, Dimension('L'))
        self.assertTrue(hard_equal(q1, q2))

        q1 = Quantity(1, Dimension('L'), symbol="toto")
        q2 = Quantity(1, Dimension('L'), symbol="toto")
        self.assertTrue(hard_equal(q1, q2))

        q1 = Quantity(1, Dimension('L'))
        q2 = Quantity(2, Dimension('L'))
        self.assertFalse(hard_equal(q1, q2))

        q1 = Quantity(1, Dimension('L'), symbol="toto")
        q2 = Quantity(1, Dimension('L'), symbol="tata")
        self.assertFalse(hard_equal(q1, q2))

    def test_very_hard_equal(self):
        q1 = Quantity(1, Dimension('L'))
        q1.favunit = V
        q2 = Quantity(1, Dimension('L'))
        q2.favunit = V
        self.assertTrue(very_hard_equal(q1, q2))

        q1 = Quantity(1, Dimension('L'), symbol="toto")
        q1.favunit = V
        q2 = Quantity(1, Dimension('L'), symbol="toto")
        q2.favunit = V
        self.assertTrue(very_hard_equal(q1, q2))

        q1 = Quantity(1, Dimension('L'), symbol="toto")
        q1.favunit = V
        q2 = Quantity(1, Dimension('L'), symbol="toto")
        q2.favunit = 1*V
        self.assertFalse(very_hard_equal(q1, q2))

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
        self.assertEqual(2 / self.x_q, Quantity(2 / self.x, 1 / self.dim))
        self.assertEqual(self.x_q / 2, Quantity(self.x / 2, self.dim))

    def test_15_abs(self):
        left = Quantity(-1, Dimension("L"))
        right = m
        self.assertTrue(abs(left) == right)
        self.assertFalse(abs(left).symbol == right.symbol)

    def test_20_test_inv(self):
        pass

    def test_30_test_favunit(self):

        #print("DISPLAY_DIGITS : " + str(DISPLAY_DIGITS))
        #print("EXP_THRESHOLD : "+ str(EXP_THRESHOLD))
        longueur = 543.21*m
        longueur.favunit = self.mm
        self.assertEqual(str(longueur), "543210.0 mm")

        cy = Quantity(1, Dimension(None), symbol="cy")
        f_cymm = 5*cy/self.mm
        f_cymm.favunit = cy/self.mm
        self.assertEqual(str(f_cymm), "5.0 cy/mm")

    def test_40_interpolateur(self):
        interp = np.interp
        # liste réels interpole liste réels
        tab_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tab_y = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        variable = 4.5
        self.assertEqual(interp(variable, tab_x, tab_y), 6.5)

        # array interpole array
        tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        variable = 4.5
        self.assertEqual(interp(variable, tab_x, tab_y), 6.5)
        # Quantité interpole array
        #m = Quantity(1,Dimension("L"),symbol="m")
        tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*m
        tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        variable = 4.5*m
        self.assertEqual(interp(variable, tab_x, tab_y), 6.5)
        # Quantite interpole quantite
        #m = Quantity(1,Dimension("L"),symbol="m")
        tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*m
        tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])*m**2
        variable = 4.5*m
        self.assertEqual(interp(variable, tab_x, tab_y), 6.5*m**2)
        # Array interpole quantite
        #m = Quantity(1,Dimension("L"),symbol="m")
        tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])*m
        variable = 4.5
        self.assertEqual(interp(variable, tab_x, tab_y), 6.5*m)
        # Interpole quantité par réel
        with self.assertRaises(DimensionError):
            #m = Quantity(1,Dimension("L"),symbol="m")
            tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*m
            tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
            variable = 4.5
            interp(variable, tab_x, tab_y)

        # Interpole array par quantité
        with self.assertRaises(DimensionError):
            #m = Quantity(1,Dimension("L"),symbol="m")
            tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
            variable = 4.5*m
            interp(variable, tab_x, tab_y)

        # Interpole Quantité avec mauvais dimension
        with self.assertRaises(DimensionError):
            #m = Quantity(1,Dimension("L"),symbol="m")
            tab_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*Quantity(1,
                                                                       Dimension(
                                                                           "J"),
                                                                       symbol="cd")
            tab_y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
            variable = 4.5*m
            interp(variable, tab_x, tab_y)

    def test_50_iterateur(self):
        u = np.array([1, 2, 3, 4, 5])*m
        u.favunit = Quantity(0.001, Dimension("L"), symbol="mm")
        self.assertEqual(u[2], 3*m)
        self.assertEqual(u[2].favunit, Quantity(
            0.001, Dimension("L"), symbol="mm"))

        # set item
        with self.assertRaises(DimensionError):
            u[4] = 35*Quantity(1, Dimension("M"))

        # scalar quantity shouldn't be iterable
        with self.assertRaises(TypeError):
            for i in 5*s:
                print("i")

    # def test_50_1_iter(self):
    #    a = 3*m
    #    with self.assertRaises(TypeError):
    #        iter(a)

    # def test_50_2_abc_iterable(self):
    #    """
    #    isintance(x, collections.abc.Iterable) is sometimes used as a way
    #    to check if x is iterable.
    #
    #    checks if x has "__iter__" attr, so we need to use getattr
    #    """
    #    import collections.abc
    #    is_iterable = np.array([1, 2, 3])*m
    #    self.assertTrue(isinstance(is_iterable, collections.abc.Iterable))
    #    is_iterable = np.array([[1, 2, 3]])*m
    #    self.assertTrue(isinstance(is_iterable, collections.abc.Iterable))
    #
    #    self.assertFalse(isinstance(m,collections.abc.Iterable))

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
        self.assertEqual(self.x_q + self.x_q,
                         Quantity(self.x*2, Dimension("L")))
        self.assertEqual(self.x_q - self.x_q, Quantity(0, Dimension("L")))

    def test_70_mul(self):
        # Scalaire
        # Q * 2 et 2 * Q
        self.assertEqual(self.x_q * 2,
                         Quantity(self.x_q.value * 2, self.x_q.dimension))
        self.assertEqual(2 * self.x_q,
                         Quantity(2 * self.x_q.value, self.x_q.dimension))
        # Q * Q
        self.assertEqual(self.x_q * self.x_q,
                         Quantity(self.x_q.value**2, self.x_q.dimension**2))
        # array
        # Q * 2 et 2 * Q
        self.assertTrue(
            np.all(self.y_q * 2 == Quantity(self.y_q.value * 2, self.y_q.dimension)))
        self.assertTrue(
            np.all(2 * self.y_q == Quantity(2 * self.y_q.value, self.y_q.dimension)))
        # Q * Q
        self.assertTrue(np.all(self.y_q * self.y_q ==
                        Quantity(self.y_q.value**2, self.y_q.dimension**2)))

        # Scalaire et array
        self.assertTrue(np.all(self.x_q * self.y_q == Quantity(self.x_q.value * self.y_q.value,
                                                               self.x_q.dimension * self.y_q.dimension)))
        self.assertTrue(np.all(self.y_q * self.x_q == Quantity(self.y_q.value * self.x_q.value,
                                                               self.y_q.dimension * self.x_q.dimension)))

    def test_75_matmul(self):
        a = np.array([1, 1, 1])*m
        b = np.array([1, 2, 3])*m

        self.assertEqual(a @ b,
                         6*m**2
                         )

        a = np.array([1, 1, 1])*m
        b = np.array([1, 2, 3])*K

        self.assertEqual(a @ b,
                         6*m*K
                         )

    def test_80_pow(self):
        # x_q
        self.assertEqual(self.x_q ** 1,
                         self.x_q)
        self.assertEqual(self.x_q ** 2,
                         Quantity(self.x_q.value**2, self.x_q.dimension**2))
        self.assertEqual(self.x_q ** (self.x_q/self.x_q),
                         self.x_q)
        # y_q
        self.assertTrue(np.all(self.y_q ** 1 == self.y_q))
        self.assertTrue(
            np.all(self.y_q ** 2 == Quantity(self.y_q.value**2, self.y_q.dimension**2)))
        with self.assertRaises(TypeError):
            # can't power by an array : the resulting dimension's
            # will not be consistents
            # this could work because the resulting power is an a
            # rray of 1, but it won't always be. Consider powering
            # 1 meter by [2 3 4].
            self.y_q ** (self.y_q/self.y_q) == self.y_q
        # z_q
        self.assertTrue(np.all(self.z_q ** 1 == self.z_q))
        self.assertTrue(
            np.all(self.z_q ** 2 == Quantity(self.z_q.value**2, self.z_q.dimension**2)))
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
        self.assertTrue(
            np.all(self.y_qs ** 2 == Quantity(self.y_qs.value**2, self.y_qs.dimension**2)))
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
        self.assertTrue(
            np.all(self.y_qu ** 2 == Quantity(self.y_qu.value**2, self.y_qu.dimension**2)))
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
        self.assertTrue(np.all(self.y_qsu ** 2 ==
                        Quantity(self.y_qsu.value**2, self.y_qsu.dimension**2)))
        with self.assertRaises(TypeError):
            self.y_qsu ** (self.y_qsu/self.y_qsu) == self.y_qsu

    def test_inverse(self):
        self.assertTrue(Quantity(1, Dimension("M")).inverse()
                        == Quantity(1, Dimension({"M": -1})))

    def test_90_getteur(self):
        # Sans unité favorite
        self.assertEqual(str(self.y_q[2]), "3.0 m")
        # Avec unité favorite
        self.assertEqual(str(self.y_qu[2]), "3000.0 mm")

    def test_120_linspace(self):

        #m = Quantity(1,Dimension("L"))
        # self.assertTrue(np.all(Quantity(np.linspace(1, 2, num=8),
        #                                Dimension('L')) == linspace(1*m, 2*m, num=8)))
        # with self.assertRaises(DimensionError):
        #    linspace(1*m, 2)

        # COMPARE TO np.linspace
        #m = Quantity(1,Dimension("L"))
        self.assertTrue(np.all(Quantity(np.linspace(1, 2, num=8),
                                        Dimension('L')) == np.linspace(1*m, 2*m, num=8)))
        with self.assertRaises(DimensionError):
            np.linspace(1*m, 2)

    def test_130_sum(self):
        # self.assertEqual(sum(cls.x_q),
        # self.assertEqual(sum(cls.y_q),
        # self.assertEqual(sum(cls.x_qs),
        # self.assertEqual(sum(cls.y_qs),
        # self.assertEqual(sum(cls.x_qu),
        # self.assertEqual(sum(cls.y_qu),
        # self.assertEqual(sum(cls.x_qsu),
        # self.assertEqual(sum(cls.y_qsu)
        pass

    def test_140_integrate(self):
        self.assertEqual(self.z_q.integrate(), Quantity(2, Dimension("L")))
        self.assertEqual(2*self.z_q.integrate(), 2*Quantity(2, Dimension("L")))

    def test_150_mean(self):
        self.assertEqual(self.z_q.mean(), Quantity(1, Dimension("L")))

    def test_151_std(self):
        self.assertEqual(self.z_q.std(), np.std(self.z_q))

    def test_152_var(self):
        self.assertEqual(self.z_q.var(), np.var(self.z_q))

    def test_160_sum(self):
        self.assertEqual(self.z_q.sum(), Quantity(3, Dimension("L")))

    def test_170_str(self):
        self.assertEqual(str(Quantity(np.array([1, 2, 3]), Dimension(None))),
                         "[1 2 3]")

    def test_180_repr(self):
        self.assertEqual(repr(Quantity(1, Dimension("L"))),
                         "<Quantity : 1 m>")
        self.assertEqual(repr(Quantity(np.array([1, 2, 3]), Dimension("L"))),
                         "<Quantity : [1 2 3] m>")

    def test_190_format(self):
        self.assertEqual("{!s}".format(Quantity(1, Dimension("L"))), "1 m")
        self.assertEqual("{!r}".format(
            Quantity(1, Dimension("L"))), "<Quantity : 1 m>")

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
        self.assertEqual(units["kg"], Quantity(1, Dimension("M")))
        self.assertEqual(units["m"], Quantity(1, Dimension("L")))
        self.assertEqual(units["s"], Quantity(1, Dimension("T")))
        self.assertEqual(units["K"], Quantity(1, Dimension("theta")))
        self.assertEqual(units["cd"], Quantity(1, Dimension("J")))
        self.assertEqual(units["A"], Quantity(1, Dimension("I")))
        self.assertEqual(units["mol"], Quantity(1, Dimension("N")))

    def test_m_mm_kg_g(self):
        self.assertEqual(units["m"], Quantity(1, Dimension("L")))
        self.assertEqual(units["mm"], Quantity(1, Dimension("L"))/1000)

        self.assertEqual(units["g"], Quantity(1, Dimension("M"))/1000)
        self.assertEqual(units["kg"], Quantity(1, Dimension("M")))
        


    def test_eq_ne(self):
        self.assertEqual(self.x_q, self.x_q)
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

    # def test_np_all(self):
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
        # with self.assertRaises(ValueError):
        #    min(self.x_q)
        # with self.assertRaises(ValueError):
        #    max(self.x_q)

        # Python's min/max
        self.assertEqual(max(self.y_q), Quantity(
            max(self.y_q.value), self.y_q.dimension))
        self.assertEqual(min(self.y_q), Quantity(
            min(self.y_q.value), self.y_q.dimension))

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
        self.assertTrue(Quantity(1, Dimension(
            "L")).has_integer_dimension_power())
        self.assertTrue(Quantity(1, Dimension(
            {"L": -2, "M": 2})).has_integer_dimension_power())
        self.assertTrue(Quantity(1, Dimension(
            None)).has_integer_dimension_power())

        self.assertFalse(Quantity(1, Dimension(
            {"L": 1.2})).has_integer_dimension_power())

    def test_units(self):
        Newton = (1 * kg) * (m * s**-2)
        self.assertEqual(units["N"], Newton)
        self.assertEqual(str(units["N"].symbol), "N")

    def test_make_quantity(self):
        q = self.x_q.__copy__()
        q.symbol = 'jojo'
        self.assertEqual(q, make_quantity(self.x_q, symbol='jojo'))
        self.assertEqual(str(q), str(make_quantity(self.x_q, symbol='jojo')))
        self.assertEqual(str(q.symbol), 'jojo')

        q = self.x_q.__copy__()
        mum = Quantity(0.000001, Dimension("L"), symbol="mum")
        q.favunit = mum
        self.assertEqual(q, make_quantity(self.x_q, favunit=mum))
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
        def func2D(y, x):
            # testing dimensions awareness
            z = y + 1*kg
            zz = x + 1*m
            return 1*kg

        self.assertAlmostEqual(
            4*kg**2*m, dblquad(func2D, 0*m, 2*m, 0*kg, 2*kg)[0])

    def test_410_exp_zero(self):
        self.assertEqual(self.x_q ** 0, 1)

    # def test_custom_units(self):
    #    from math import pi
    #   self.assertEqual(custom_units['deg'], Quantity(pi /180, Dimension('rad')))
    def test_trigo(self):
        #print(np.cos(self.x_q/self.x_q * 0))
        self.assertEqual(1, np.cos(self.x_q/self.x_q * 0))
        with self.assertRaises(DimensionError):
            np.cos(self.x_q)

    def test_300_define_fraction(self):
        self.assertEqual(Fraction(1, 2) * m,
                         Quantity(Fraction(1, 2), Dimension("L")))

    def test_310_fraction_operation(self):
        self.assertEqual(Fraction(1, 2) * m * 2, Fraction(1, 1) * m)
        self.assertEqual(Fraction(1, 2) * m + Fraction(1, 2)
                         * m, Fraction(1, 1) * m)
        self.assertEqual(Fraction(1, 2) * m / 2, Fraction(1, 4) * m)
        self.assertEqual(2 / (Fraction(1, 2) * m), 4 * 1/m)
        with self.assertRaises(DimensionError):
            Fraction(1, 2) * m + 1
        self.assertTrue(Fraction(1, 2) * m <= Fraction(3, 2) * m)

    def test_400_complex(self):
        self.assertEqual((1j+1) * m, Quantity((1j+1), Dimension("L")))
        self.assertEqual((1j+1) * m + 1 * m, Quantity((1j+2), Dimension("L")))
        self.assertEqual((2j+4) * m + (5j-1) * m,
                         Quantity((7j+3), Dimension("L")))

    def test_pos_neg(self):
        self.assertEqual(+m, m)
        self.assertTrue(
            np.all(+np.array([1, 2, 3])*m == np.array([1, 2, 3])*m))
        self.assertEqual(-m, Quantity(-1, Dimension("L")))

    def test_500_numpy_ufuncs(self):

        arr = np.array([1, 2, 3])
        arr_m = Quantity(arr, Dimension("L"))

        # add
        self.assertTrue(np.all(m + arr_m == Quantity(1 + arr, Dimension("L"))))
        self.assertTrue(np.all(arr_m + m == Quantity(1 + arr, Dimension("L"))))
        self.assertTrue(np.all(np.add(arr_m, arr_m) ==
                        Quantity(2 * arr, Dimension("L"))))

        # sub
        self.assertTrue(np.all(m - arr_m == Quantity(1 - arr, Dimension("L"))))
        self.assertTrue(np.all(arr_m - m == Quantity(arr - 1, Dimension("L"))))

        self.assertTrue(np.all(np.subtract(m, arr_m) ==
                        Quantity(1 - arr, Dimension("L"))))
        self.assertTrue(np.all(np.subtract(arr_m, m) ==
                        Quantity(arr - 1, Dimension("L"))))
        self.assertTrue(np.all(np.subtract(arr_m, arr_m) ==
                        Quantity(0 * arr, Dimension("L"))))

        # mul
        self.assertTrue(
            np.all(m * arr_m == Quantity(1 * arr, Dimension({"L": 2}))))
        self.assertTrue(
            np.all(arr_m * m == Quantity(arr * 1, Dimension({"L": 2}))))

        self.assertTrue(np.all(np.multiply(m, arr_m) ==
                        Quantity(1 * arr, Dimension({"L": 2}))))
        self.assertTrue(np.all(np.multiply(arr_m, m) ==
                        Quantity(arr * 1, Dimension({"L": 2}))))
        self.assertTrue(np.all(np.multiply(arr_m, arr_m) ==
                        Quantity(arr * arr, Dimension({"L": 2}))))

        # matmul
        self.assertTrue(np.all(np.matmul(arr_m, arr_m) ==
                        Quantity(arr @ arr, Dimension({"L": 2}))))

        # div
        self.assertTrue(np.all(m / arr_m == np.array([1/1, 1/2, 1/3])))
        self.assertTrue(np.all(arr_m / m == np.array([1., 2., 3.])))

        self.assertTrue(np.all(np.divide(m, arr_m) ==
                        np.array([1/1, 1/2, 1/3])))
        self.assertTrue(np.all(np.divide(arr_m, m) == np.array([1., 2., 3.])))
        self.assertTrue(np.all(np.divide(arr_m, arr_m)
                        == np.array([1., 1., 1.])))

        # logaddexp
        self.assertTrue(np.logaddexp(1, 2) == np.logaddexp(
            Quantity(1, Dimension(None)), Quantity(2, Dimension(None))))

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
        self.assertTrue(np.negative(Quantity(3, Dimension(None)))
                        == np.negative(3)*Quantity(1, Dimension(None)))

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
        self.assertTrue(
            np.all(np.fmod([5, 3]*m, [2, 2.]*m) == np.array([1, 1])*m))

        # floor
        self.assertTrue(np.floor(3.4*m) == np.floor(3.4)*m)

        # ceil
        self.assertTrue(np.ceil(3.4*m) == np.ceil(3.4)*m)

        # trunc
        self.assertTrue(np.trunc(3.4*m) == np.trunc(3.4)*m)

        # absolute
        self.assertEqual(np.absolute(5*m), 5*m)
        self.assertEqual(np.absolute(-5*m), 5*m)
        self.assertTrue(np.all(np.absolute(np.arange(-5, 5)*m)
                        == np.array([5, 4, 3, 2, 1, 0, 1, 2, 3, 4])*m))

        # rint
        self.assertEqual(np.rint(5.3*m), 5*m)
        self.assertTrue(
            np.all(np.rint(np.array([1.1, 2.3, 3.5, 4.6])*m) == np.array([1, 2, 4, 5])*m))

        # sign
        self.assertEqual(np.sign(-3*m), -1)
        self.assertEqual(np.sign(3*m), 1)

        self.assertTrue(
            np.all(np.sign(np.array([-1, 0, 1])*m) == np.array([-1, 0, 1])))

        # conj and conjugate : conj is an alias for conjugate
        self.assertEqual(np.conj(3*m), 3*m)
        self.assertEqual(np.conj(3*m+2j*m), 3*m-2j*m)
        self.assertEqual(np.conjugate(3*m), 3*m)
        self.assertEqual(np.conjugate(3*m+2j*m), 3*m-2j*m)
        self.assertTrue(
            np.all(np.conj(np.arange(3)*m+1j*m) == np.arange(3)*m-1j*m))
        self.assertTrue(
            np.all(np.conjugate(np.arange(3)*m+1j*m) == np.arange(3)*m-1j*m))

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
        self.assertTrue(
            np.all(np.square(np.arange(3)*m) == (np.arange(3)*m)**2))

        # cbrt
        self.assertEqual(np.cbrt((3*m)),
                         (3*m)**(1/3))
        self.assertTrue(np.allclose(
            np.cbrt((np.arange(3)*m)).value, ((np.arange(3)*m)**(1/3)).value))

        # reciprocal
        self.assertEqual(np.reciprocal(3.1*m),
                         1/(3.1*m))
        self.assertTrue(np.all(np.reciprocal(np.array([1.1, 2.3])*m) ==
                        1/(np.array([1.1, 2.3])*m)))

        # pow
        with self.assertRaises(TypeError):
            np.power(m, arr_m)
        with self.assertRaises(TypeError):
            np.power(arr_m, m)
        self.assertTrue(np.all(arr_m ** 1 == arr_m))
        self.assertTrue(np.all(arr_m ** 2 == arr_m * arr_m))

        # hypot
        self.assertTrue(
            np.all(np.hypot(m, m) == Quantity((1+1)**(1/2), Dimension("L"))))
        self.assertTrue(np.all(np.hypot(m, arr_m) == Quantity(
            np.hypot(1, np.array([1, 2, 3])), Dimension("L"))))
        self.assertTrue(np.all(np.hypot(arr_m, m) == np.hypot(m, arr_m)))

        # greater
        self.assertTrue(np.all(np.greater(m, m) == False))
        self.assertTrue(np.all(np.greater(m, arr_m) ==
                        np.array([False, False, False])))
        self.assertTrue(np.all(np.greater(arr_m, m) ==
                        np.array([False, True, True])))
        self.assertTrue(np.all(np.greater(arr_m, arr_m) ==
                        np.array([False, False, False])))

        # greater_or_equal
        self.assertTrue(np.all(np.greater_equal(m, m) == True))
        self.assertTrue(np.all(np.greater_equal(m, arr_m)
                        == np.array([True, False, False])))
        self.assertTrue(np.all(np.greater_equal(arr_m, m)
                        == np.array([True, True, True])))
        self.assertTrue(np.all(np.greater_equal(arr_m, arr_m)
                        == np.array([True, True, True])))

        # less
        self.assertTrue(np.all(np.less(m, m) == False))
        self.assertTrue(np.all(np.less(m, arr_m) ==
                        np.array([False, True, True])))
        self.assertTrue(np.all(np.less(arr_m, m) ==
                        np.array([False, False, False])))
        self.assertTrue(np.all(np.less(arr_m, arr_m) ==
                        np.array([False, False, False])))

        # less_or_equal
        self.assertTrue(np.all(np.less_equal(m, m) == True))
        self.assertTrue(np.all(np.less_equal(m, arr_m)
                        == np.array([True, True, True])))
        self.assertTrue(np.all(np.less_equal(arr_m, m) ==
                        np.array([True, False, False])))
        self.assertTrue(np.all(np.less_equal(arr_m, arr_m)
                        == np.array([True, True, True])))

        # equal
        self.assertTrue(np.all(np.equal(arr_m, arr_m)))
        self.assertTrue(np.all(np.equal(m, m)))

        # not_equal
        self.assertTrue(np.all(np.not_equal(arr_m, 0*m)))
        self.assertTrue(np.all(np.not_equal(m, 0*m)))

        # isreal
        # self.assertTrue(np.isreal(m))
        # self.assertTrue(np.all(np.isreal(arr_m)))

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

        self.assertTrue(np.all(np.copysign(m, arr_m) == np.array([1, 1, 1])*m))
        self.assertTrue(np.all(np.copysign(m, -arr_m)
                        == -np.array([1, 1, 1])*m))

        # nextafter
        eps = np.finfo(np.float64).eps
        self.assertTrue(np.nextafter(1*m, 2*m) == eps*m+1*m)

        # modf
        res_modf = np.modf([0, 3.5]*m)
        frac, integ = res_modf
        self.assertTrue(np.all(frac == np.array([0, 0.5])*m))
        self.assertTrue(np.all(integ == np.array([0, 3])*m))

        # sqrt
        self.assertEqual(np.sqrt(m), Quantity(1, Dimension({"L": 1/2})))
        self.assertTrue(np.all(np.sqrt(arr_m) == Quantity(
            np.sqrt(np.array([1., 2., 3.])), Dimension({"L": 1/2}))))

        # Trigo
        zero_rad = Quantity(0, Dimension("RAD"))
        zero_none = Quantity(0, Dimension(None))
        one_none = Quantity(1, Dimension(None))
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
        self.assertTrue(np.arctan2(m, m) == np.arctan2(1, 1))
        with self.assertRaises(DimensionError):
            np.arctan2(m, 1)
        with self.assertRaises(DimensionError):
            np.arctan2(1, m)
        self.assertTrue(np.arctan2(one_none, 1) == np.arctan2(1, 1))
        self.assertTrue(np.arctan2(1, one_none) == np.arctan2(1, 1))

        # fabs
        self.assertEqual(np.fabs(m), m)
        self.assertEqual(np.fabs(-m), m)
        self.assertTrue(np.all(np.fabs(arr_m) == arr_m))
        self.assertTrue(np.all(np.fabs(-arr_m) == arr_m))
        self.assertTrue(np.all(np.fabs(Quantity(np.array(
            [-1, 0, 1]), Dimension("L"))) == Quantity(np.array([1, 0, 1]), Dimension("L"))))

    def test_510_numpy_functions(self):

        arr = np.array([1, 2, 3])
        arr_m = Quantity(arr, Dimension("L"))

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
        self.assertTrue(np.all(np.append(
            Quantity(1, Dimension(None)), 1) == Quantity([1, 1], Dimension(None))))

        # np.argmax
        self.assertEqual(np.argmax(np.arange(5)*m), 4*m)

        # np.argmin
        self.assertEqual(np.argmin(np.arange(5)*m), 0*m)

        # np.argsort
        self.assertTrue(np.all(np.argsort(np.arange(5)*m) == np.arange(5)))

        # np.sort
        self.assertTrue(np.all(np.sort(np.arange(5)*m) == np.arange(5)*m))

        # np.around
        self.assertTrue(np.all(np.around(np.linspace(2.2, 3.656, 10)*m)
                        == np.around(np.linspace(2.2, 3.656, 10))*m))

        # np.atleast_1d
        self.assertTrue(np.all(np.atleast_1d(4*m) == np.array([4])*m))
        self.assertTrue(np.all(np.atleast_1d(
            np.arange(3)*m) == np.arange(3)*m))
        left_in = np.arange(5)*m
        right_in = 4*m
        res_in1, res_in2 = np.atleast_1d(left_in, right_in)
        self.assertTrue(np.all(res_in1 == np.arange(5)*m))
        self.assertTrue(np.all(res_in2 == np.array([4])*m))

        # np.average
        self.assertEqual(np.average(arr_m), 2*m)
        self.assertEqual(np.average(5*m), 5*m)

        # np.broadcast_arrays
        x = np.array([[1, 2, 3]])*m
        y = np.array([[4], [5]])*K
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

        x = np.array([[1, 2, 3]])*m
        y = np.array([[4], [5]])
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
        # res = np.block([
        #    [A,               np.zeros((2, 3))*m],
        #    [np.ones((3, 2))*m, B*m               ]
        # ])
        # exp = array(
        #    [[2., 0., 0., 0., 0.],
        #    [0., 2., 0., 0., 0.],
        #    [1., 1., 3., 0., 0.],
        #    [1., 1., 0., 3., 0.],
        #    [1., 1., 0., 0., 3.]
        #    ])*m
        #self.assertTrue(np.all(res, exp))

        # np.copyto
        a = np.array([1, 1, 1])*m
        b = np.array([1, 2, 3])*m
        np.copyto(a, b)
        self.assertTrue(np.all(a == np.array([1, 2, 3])*m))

        with self.assertRaises(DimensionError):
            a = np.array([1, 1, 1])*m
            b = np.array([1, 2, 3])*s
            np.copyto(a, b)

        with self.assertRaises(DimensionError):
            a = np.array([1, 1, 1])
            b = np.array([1, 2, 3])*s
            np.copyto(a, b)

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
        self.assertTrue(np.all(np.column_stack((a, b)) == np.array([[1, 2],
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
        self.assertTrue(np.all(np.concatenate((a, b), axis=0) == np.array([[1, 2],
                                                                           [3, 4],
                                                                           [5, 6]])*m))

        # np.copy
        self.assertTrue(np.all(np.copy(np.arange(3)*m) == np.arange(3)*m))

        # np.cross
        self.assertTrue(np.all(np.cross(np.array([[1, 2, 3]])*m,
                                        np.array([[4, 5, 6]])*m) == np.array([-3, 6, -3])*m**2))

        # np.cumprod

        # np.cumsum
        self.assertTrue(np.all(np.cumsum(np.arange(3)*m)
                        == np.array([0, 1, 3])*m))

        # np.diagonal
        a = np.arange(4).reshape(2, 2)*m
        self.assertTrue(np.all(np.diagonal(a) == np.array([0, 3])*m))

        # np.diff
        a = np.array([1, 2, 4, 7, 0])*m
        self.assertTrue(np.all(np.diff(a) == np.array([1, 2, 3, -7])*m))

        # np.dot
        a = np.array([[1, 0], [0, 1]])*m
        b = np.array([[4, 1], [2, 2]])*m
        self.assertTrue(np.all(np.dot(a, b) == np.array([[4, 1],
                                                         [2, 2]])*m**2))

        # np.dstack
        a = np.array((1, 2, 3))*m
        b = np.array((2, 3, 4))*m
        self.assertTrue(np.all(np.dstack((a, b)) == np.array([[1, 2],
                                                              [2, 3],
                                                              [3, 4]])*m))

        # np.ediff1d

        # np.histogram
        arr = np.random.normal(1, 2)
        arrq = arr*m
        hist_q, bins_q = np.histogram(arrq)
        hist, bins = np.histogram(arr)
        self.assertTrue(np.all(hist_q == hist))
        self.assertTrue(np.all(bins_q == bins*m))

        # np.sum
        self.assertEqual(np.sum(arr_m), 6 * m)
        self.assertEqual(np.sum(5*m), 5 * m)
        self.assertTrue(np.all(np.sum(np.array([[1, 2, 3],
                                                [1, 2, 3]])*m, axis=1) == np.array([6, 6])*m))

        # np.mean
        self.assertEqual(np.mean(arr_m), 2*m)
        self.assertEqual(np.mean(5*m), 5*m)
        self.assertTrue(np.all(np.mean(np.arange(6).reshape(
            3, 2)*s, axis=1) == np.mean(np.arange(6).reshape(3, 2), axis=1)*s))

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
        # q array, dx quantity --> see issue on GH
        # self.assertEqual(np.trapz(np.arange(5), dx=1*m),
        #                 np.trapz(np.arange(5), dx=1)*m)
        # q array, x quantity
        self.assertEqual(np.trapz(np.arange(5), x=np.arange(5)*m),
                         np.trapz(np.arange(5), x=np.arange(5))*m)

        # np.linspace
        self.assertTrue(np.all(np.linspace(0*m, 5*m) ==
                        Quantity(np.linspace(0, 5), Dimension("L"))))

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

    def test_np_linalg_inv(self):

        ohm = units["ohm"]
        volt = units["V"]

        R = np.array([[50, 0, -30],
                      [0, 40, -20],
                      [-30, -20, 100]])*ohm
        V = np.array([80, 80, 0]) * volt
        I = np.linalg.inv(R) @ V
        self.assertTrue(I.dimension == Dimension("A"))
        self.assertTrue(np.all(I.value == np.linalg.inv(R.value)@V.value))

    def test_real(self):
        self.assertEqual(m.real, 1*m)
        a = np.array([1, 2, 3])*m
        self.assertTrue(np.all(a.real == np.array([1, 2, 3])*m))

    def test_np_real(self):
        self.assertEqual(np.real(m), m)
        self.assertTrue(
            np.all(np.real(np.array([1, 2, 3])*m) == np.array([1, 2, 3])*m))
        self.assertTrue(
            np.all(np.real(np.array([1+0j, 2+0j, 3+0j])*m) == np.array([1, 2, 3])*m))

    def test_imag(self):
        self.assertEqual(m.imag, 0*m)
        a = np.array([1, 2, 3])*m
        self.assertTrue(np.all(a.imag == np.array([0, 0, 0])*m))

    def test_transpose(self):
        a = np.array([[1, 0], [0, 1]])*m
        self.assertTrue(np.all(a.T == np.array([[1, 0], [0, 1]]).T*m))

    def test_transpose_fail(self):
        #m = Quantity(1, Dimension("L"))
        with self.assertRaises(AttributeError):
            q_test = 5*m
            q_test.T

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
        self.assertTrue(
            np.all(sum([self.x_q, self.y_q], 0*m) == sum([self.x, self.y])*m))
        with self.assertRaises(DimensionError):
            sum([self.x_q, self.y_q])

    def test_500_decorator_check_dimension(self):

        # To check the dimension analysis of inputs
        # Two inputs, one output
        def speed(l, t):
            return l/t
        wrapped_speed = check_dimension((m, s), m/s)(speed)

        with self.assertRaises(DimensionError):
            wrapped_speed(1, 1)  # a scalar is interpreted as dimensionless
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
            wrapped_speed(1, 1)  # a scalar is interpreted as dimensionless
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
        self.assertEqual(dimension_and_favunit((km, s), mph)
                         (speed)(5*m, 2*s), speed(5*m, 2*s))

        # favunit
        self.assertEqual(dimension_and_favunit((km, s), mph)
                         (speed)(5*m, 2*s).favunit, mph)

        # dimension check
        with self.assertRaises(DimensionError):
            dimension_and_favunit((km, s), mph)(speed)(5*s, 2*s)

    def test_503_decorator_drop_dimension(self):
        # this function will always compare inputs to ints
        # so the inputs must be scalar of dimless Quantitys
        def speed_dimless(l, t):
            if not t == 0 and not l < 0:
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
            if not t == 0 and not l < 0:
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
            asqarray(np.array([1*s, 2*s], dtype=object)
                     ) == Quantity([1, 2], Dimension("T"))
        ))

        self.assertTrue(np.all(
            asqarray(np.array([1*m], dtype=object)
                     ) == Quantity([1], Dimension("L"))
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

        arrq_9 = np.array([m.__copy__()], dtype=object)
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

    def test_math_ceil(self):
        a = 5.123*m
        self.assertEqual(math.ceil(a), math.ceil(5.123)*m)

    def test_math_floor(self):
        a = 5.123*m
        self.assertEqual(math.floor(a), math.floor(5.123)*m)

    def test_math_trunc(self):
        a = 5.123*m
        self.assertEqual(math.trunc(a), math.trunc(5.123)*m)

    def test_check_dim(self):
        self.assertTrue(m.check_dim(Dimension("L")))
        self.assertFalse(m.check_dim(Dimension("RAD")))

    def test_matplotlib(self):
        # make sure units are disabled
        setup_matplotlib(enable=False)

        arr_m = np.linspace(1, 3, 2)*1000 * m
        arr_s = np.linspace(1, 3, 2)*1000 * s

        fig, ax = matplotlib.pyplot.subplots()

        # very permissive, dimension are simply dropped
        ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        ax.plot(arr_m, np.linspace(1, 3, 2), "o")
        ax.plot(arr_s, arr_m)
        ax.plot(arr_m, arr_s)

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
        # self.assertEqual(ax.yaxis.units,
        #                km_per_sr)
        # self.assertEqual(ax.yaxis.label.get_text(),
        #                 "km/sr")

        # plot with non unitary, different dimension
        #two_km_per_sr = 2*km/sr
        #two_km_per_sr.symbol = "2km/sr"
        #arr_m.favunit = two_km_per_sr
        #fig, ax = matplotlib.pyplot.subplots()
        #ax.plot(np.linspace(1, 3, 2), arr_m, "o")
        # self.assertEqual(ax.yaxis.units,
        #                two_km_per_sr)
        # self.assertEqual(ax.yaxis.label.get_text(),
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
        ax.plot(np.linspace(1, 3, 2), Quantity(
            np.linspace(1, 3, 2), Dimension(None), "o"))
        with self.assertRaises(DimensionError):
            ax.plot(np.linspace(1, 3, 2), arr_m**2, "o")

    def test_matplotlib_axhlines(self):
        # was not working due to np.iterable(3*m) returning True
        with plotting_context():
            y = np.linspace(0, 30) * mm
            x = np.linspace(0, 5) * s

            fig, ax = plt.subplots()
            ax.plot(x, y, 'tab:blue')
            ax.axhline(0.02 * m, color='tab:red')

    def test_matplotlib_axvlines(self):
        # was not working due to np.iterable(3*m) returning True
        with plotting_context():
            y = np.linspace(0, 30) * mm
            x = np.linspace(0, 5) * s

            fig, ax = plt.subplots()
            ax.plot(x, y, 'tab:blue')
            ax.axvline(0.02 * s, color='tab:red')

    def test_matplotlib_twinx(self):
        with plotting_context():

            fig, ax = plt.subplots()
            ax.plot(asqarray([m, 2*m]),
                    asqarray([2*m, 3*m]), "-o")
            ax2 = ax.twinx()
            ax2.plot(m, 3*s, "*", color="r")

    def test_matplotlib_favunit_volts(self):
        with plotting_context():

            fig, ax = plt.subplots()
            ax.plot(3*m, 5*V, "-o")
            self.assertEqual(ax.yaxis.units,
                             V)
            self.assertEqual(ax.yaxis.label.get_text(),
                             "V")

    def test_matplotlib_quickplot(self):
        y = np.linspace(0, 30) * mm
        y.plot()

    def test_matplotlib_scatter_masked(self):
        with plotting_context():
            secs = units["s"]
            hertz = units["Hz"]
            minutes = units["min"]

            # create masked array
            data = (1, 2, 3, 4, 5, 6, 7, 8)
            mask = (1, 0, 1, 0, 0, 0, 1, 0)
            xsecs = secs * np.ma.MaskedArray(data, mask, float)

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

            ax1.scatter(xsecs, xsecs)
            ax2.scatter(xsecs, 1/xsecs, yunits=hertz)
            ax3.scatter(xsecs, xsecs, yunits=minutes)

            self.assertTrue(ax1.yaxis.units == s)
            self.assertTrue(ax2.yaxis.units == hertz)
            self.assertTrue(ax3.yaxis.units == minutes)

    def test_matplotlib_set_limits_on_blank_plot(self):
        with plotting_context():
            from physipy import units, s, imperial_units, setup_matplotlib, m
            inch = imperial_units["in"]
            fig, ax = plt.subplots()
            ax.set_xlim(2*s, 3*s)
            ax.set_ylim(1*inch, 7*inch)
            self.assertTrue(ax.xaxis.units == s)
            self.assertTrue(ax.yaxis.units == m)

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
        arr = np.arange(10).reshape(5, 2)
        qarr = arr*m

        self.assertTrue(np.all(
            qarr.flatten() == np.arange(10)*m
        ))

    def test_flatten_raises(self):
        qflat = 5*m
        with self.assertRaises(AttributeError):
            qflat.flatten()

    def test_ravel(self):

        exp = np.array([1, 2, 3, 4, 5, 6])*m
        res = np.ravel(np.array([[1, 2, 3], [4, 5, 6]])*m)
        self.assertTrue(np.all(exp == res))

    def test_reshape(self):
        exp = np.array([1, 2, 3, 4, 5, 6])*m
        res = np.reshape(np.array([[1, 2, 3], [4, 5, 6]])*m, (1, 6))
        self.assertTrue(np.all(exp == res))

    def test_xvectorize(self):
        # func returns dimensionfull value
        arr_m = np.arange(5)*m

        def thresh(x):
            if x > 3*m:
                return x
            else:
                return 3*m
        vec_thresh = xvectorize(thresh)

        res = vec_thresh(arr_m)
        exp = np.array([3, 3, 3, 3, 4])*m
        self.assertTrue(np.all(res == exp))

        # func returns dimensionless value
        arr_m = np.arange(5)*m

        def thresh(x):
            if x > 3*m:
                return x/m
            else:
                return 3
        vec_thresh = xvectorize(thresh)

        res = vec_thresh(arr_m)
        exp = np.array([3, 3, 3, 3, 4])
        self.assertTrue(np.all(res == exp))

    # def test_vectorize(self):
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
        # func returns dimensionfull value
        arr_m = np.arange(6).reshape(3, 2)*m

        def thresh(x):
            if x > 3*m:
                return x
            else:
                return 3*m
        vec_thresh = ndvectorize(thresh)

        res = vec_thresh(arr_m)
        exp = np.array([[3, 3], [3, 3], [4, 5]])*m
        self.assertTrue(np.all(res == exp))

        # func returns dimensionless value
        arr_m = np.arange(6).reshape(3, 2)*m

        def thresh(x):
            if x > 3*m:
                return x/m
            else:
                return 3
        vec_thresh = ndvectorize(thresh)

        res = vec_thresh(arr_m)
        exp = np.array([[3, 3], [3, 3], [4, 5]])
        self.assertTrue(np.all(res == exp))

    def test_np_fft_fftshift(self):
        exp = np.fft.fftshift(np.arange(10))*s
        self.assertTrue(np.all(np.fft.fftshift(np.arange(10)*s) == exp))

    def test_np_fft_ifftshift(self):
        exp = np.fft.ifftshift(np.arange(10))*s
        self.assertTrue(np.all(np.fft.ifftshift(np.arange(10)*s) == exp))

    def test_np_fft_fft(self):
        exp = np.fft.fft(np.arange(10))*s
        self.assertTrue(np.all(np.fft.fft(np.arange(10)*s) == exp))

    def test_np_fft_ifft(self):
        exp = np.fft.ifft(np.arange(10))*s
        self.assertTrue(np.all(np.fft.ifft(np.arange(10)*s) == exp))

    def test_qarange(self):
        self.assertTrue(np.all(qarange(0*s, 1*s, 0.1*s)
                        == np.arange(0, 1, 0.1)*s))
        self.assertTrue(np.all(qarange(0*s, 0.1*s) == np.arange(0, 0.1)*s))
        self.assertTrue(np.all(qarange(0*s, step=0.1*s)
                        == np.arange(0, step=0.1)*s))
        self.assertTrue(np.all(qarange(0*s, stop=0.1*s)
                        == np.arange(0, stop=0.1)*s))
        self.assertTrue(np.all(qarange(0*s, step=0.1*s,
                                       stop=2*s) == np.arange(0, step=0.1, stop=2)*s))

    def test_np_convolve(self):
        arr_s = np.ones(10)*s
        arr = np.ones(10)

        self.assertTrue(np.all(np.convolve(arr_s, arr) ==
                        np.convolve(np.ones(10), np.ones(10))*s))
        self.assertTrue(np.all(np.convolve(arr_s, arr_s) ==
                        np.convolve(np.ones(10), np.ones(10))*s**2))

    def test_np_vstack(self):
        a = np.array([[1, 1, 1],
                      [1, 3, 4],
                      [4, 6, 1]])*m
        b = np.array([1, 0.5, 3])*m
        c = np.array([1, 0.5, 3])*K

        self.assertTrue(np.all(np.vstack((a, a)) ==
                        np.vstack((a.value, a.value))*m))
        self.assertTrue(np.all(np.vstack((a, b)) ==
                        np.vstack((a.value, b.value))*m))

        with self.assertRaises(DimensionError):
            np.vstack((a, c))

    def test_np_hstack(self):
        a = np.array([[1, 1, 1],
                      [1, 3, 4],
                      [4, 6, 1]])*m
        b = np.array([[1], [0.5], [3]])*m
        c = np.array([[1], [0.5], [3]])*K

        self.assertTrue(np.all(np.hstack((a, a)) ==
                        np.hstack((a.value, a.value))*m))
        self.assertTrue(np.all(np.hstack((a, b)) ==
                        np.hstack((a.value, b.value))*m))

        with self.assertRaises(DimensionError):
            np.hstack((a, c))

    def test_np_stack(self):
        a = np.array([[1], [0.5], [3]])*m
        b = np.array([[1], [0.5], [3]])*m
        c = np.array([[1], [0.5], [3]])*K

        self.assertTrue(np.all(np.stack((a, a)) ==
                        np.stack((a.value, a.value))*m))
        self.assertTrue(np.all(np.stack((a, b)) ==
                        np.stack((a.value, b.value))*m))

        with self.assertRaises(DimensionError):
            np.stack((a, c))

    def test_np_roll(self):
        res = np.roll(np.arange(10)*m, 2)
        exp = [8, 9, 0, 1, 2, 3, 4, 5, 6, 7]*m
        self.assertTrue(np.all(res == exp))

    def test_np_add_reduce(self):
        res = np.add.reduce(np.arange(10)*m)
        exp = np.add.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_add_accumulate(self):
        exp = np.add.accumulate(np.arange(10))*m
        res = np.add.accumulate(np.arange(10)*m)
        self.assertTrue(np.all(res == exp))

    def test_np_subtract_reduce(self):
        res = np.subtract.reduce(np.arange(10)*m)
        exp = np.subtract.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_maximum_reduce(self):
        res = np.maximum.reduce(np.arange(10)*m)
        exp = np.maximum.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_minimum_reduce(self):
        res = np.minimum.reduce(np.arange(10)*m)
        exp = np.minimum.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_fmax_reduce(self):
        res = np.fmax.reduce(np.arange(10)*m)
        exp = np.fmax.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_fmin_reduce(self):
        res = np.fmin.reduce(np.arange(10)*m)
        exp = np.fmin.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_remainder_reduce(self):
        res = np.remainder.reduce(np.arange(10)*m)
        exp = np.remainder.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_mod_reduce(self):
        res = np.mod.reduce(np.arange(10)*m)
        exp = np.mod.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_fmod_reduce(self):
        res = np.fmod.reduce(np.arange(10)*m)
        exp = np.fmod.reduce(np.arange(10))*m
        self.assertEqual(res, exp)

    def test_np_multiply_reduce(self):
        res = np.multiply.reduce(np.arange(10)*m)
        exp = np.multiply.reduce(np.arange(10))*m**10
        self.assertEqual(res, exp)

    def test_np_greater_reduce(self):
        arr = np.arange(10)

        # with dtype bool
        exp = np.greater.reduce(arr, dtype=bool)
        res = np.greater.reduce(arr*m, dtype=bool)
        self.assertEqual(res, exp)

        # as object
        exp = np.greater.reduce(arr.astype(object))
        res = np.greater.reduce(arr.astype(object)*m)
        self.assertEqual(res, exp)

        # see https://github.com/numpy/numpy/issues/20929
        # and https://github.com/numpy/numpy/pull/22223
        # exp = np.greater.reduce(arr)
        # res = np.greater.reduce(arr*m)
        # self.assertEqual(res, exp)

    def test_np_greater_equal_reduce(self):
        arr = np.arange(10)

        # with dtype bool
        exp = np.greater_equal.reduce(arr, dtype=bool)
        res = np.greater_equal.reduce(arr*m, dtype=bool)
        self.assertEqual(res, exp)

        # as object
        exp = np.greater_equal.reduce(arr.astype(object))
        res = np.greater_equal.reduce(arr.astype(object)*m)
        self.assertEqual(res, exp)

        # see https://github.com/numpy/numpy/issues/20929
        # and https://github.com/numpy/numpy/pull/22223
        # exp = np.greater_equal.reduce(arr)
        # res = np.greater_equal.reduce(arr*m)
        # self.assertEqual(res, exp)

    def test_np_less_reduce(self):

        arr = np.arange(10)

        # with dtype bool
        exp = np.less.reduce(arr, dtype=bool)
        res = np.less.reduce(arr*m, dtype=bool)
        self.assertEqual(res, exp)

        # as object
        exp = np.less.reduce(arr.astype(object))
        res = np.less.reduce(arr.astype(object)*m)
        self.assertEqual(res, exp)

        # see https://github.com/numpy/numpy/issues/20929
        # and https://github.com/numpy/numpy/pull/22223
        # exp = np.less.reduce(arr)
        # res = np.less.reduce(arr*m)
        # self.assertEqual(res, exp)

    def test_np_less_equal_reduce(self):

        arr = np.arange(10)

        # with dtype bool
        exp = np.less_equal.reduce(arr, dtype=bool)
        res = np.less_equal.reduce(arr*m, dtype=bool)
        self.assertEqual(res, exp)

        # as object
        exp = np.less_equal.reduce(arr.astype(object))
        res = np.less_equal.reduce(arr.astype(object)*m)
        self.assertEqual(res, exp)

        # see https://github.com/numpy/numpy/issues/20929
        # and https://github.com/numpy/numpy/pull/22223
        # exp = np.less_equal.reduce(arr)
        # res = np.less_equal.reduce(arr*m)
        # self.assertEqual(res, exp)

    def test_np_not_equal_reduce(self):

        arr = np.arange(10)

        # with dtype bool
        exp = np.not_equal.reduce(arr, dtype=bool)
        res = np.not_equal.reduce(arr*m, dtype=bool)
        self.assertEqual(res, exp)

        # as object
        exp = np.not_equal.reduce(arr.astype(object))
        res = np.not_equal.reduce(arr.astype(object)*m)
        self.assertEqual(res, exp)

        # see https://github.com/numpy/numpy/issues/20929
        # and https://github.com/numpy/numpy/pull/22223
        # exp = np.not_equal.reduce(arr)
        # res = np.not_equal.reduce(arr*m)
        # self.assertEqual(res, exp)

    def test_np_equal_reduce(self):
        arr = np.arange(10)

        # with dtype bool
        exp = np.equal.reduce(arr, dtype=bool)
        res = np.equal.reduce(arr*m, dtype=bool)
        self.assertEqual(res, exp)

        # as object
        exp = np.equal.reduce(arr.astype(object))
        res = np.equal.reduce(arr.astype(object)*m)
        self.assertEqual(res, exp)

        # see https://github.com/numpy/numpy/issues/20929
        # and https://github.com/numpy/numpy/pull/22223
        # exp = np.equal.reduce(arr)
        # res = np.equal.reduce(arr*m)
        # self.assertEqual(res, exp)

    def test_np_min_max(self):
        res = np.max(np.arange(10)*m)
        exp = np.max(np.arange(10))*m
        self.assertTrue(np.all(res == exp))

        res = np.min(np.arange(10)*m)
        exp = np.min(np.arange(10))*m
        self.assertTrue(np.all(res == exp))

    def test_np_floor_dividel_reduce(self):
        res = np.floor_divide.reduce(np.arange(10)*m)
        exp = np.floor_divide.reduce(np.arange(10))
        self.assertEqual(res, exp)

    def test_np_stride_sliding_window(self):
        arr = np.arange(100).reshape(10, 10)
        res = np.lib.stride_tricks.sliding_window_view(arr*m, (4, 4))
        exp = Quantity(np.lib.stride_tricks.sliding_window_view(
            arr, (4, 4)), Dimension("L"))
        self.assertTrue(np.all(res == exp))

    def test_np_count_nonzero(self):
        exp = np.count_nonzero(np.arange(10))
        res = np.count_nonzero(np.arange(10)*m)
        self.assertEqual(exp, res)

    def test_np_gradient(self):
        exp = Quantity(np.gradient(np.arange(10), 0.5), Dimension("L"))
        res = np.gradient(np.arange(10)*m, 0.5)
        self.assertTrue(np.all(res == exp))

        # meters should simplify and return a plain arrays
        exp = np.gradient(np.arange(10), 0.5)
        res = np.gradient(np.arange(10)*m, 0.5*m)
        self.assertTrue(np.all(res == exp))

    def test_np_insert(self):
        exp = np.array([1, 2, 3, 4])*m
        res = np.insert(np.array([1, 2, 4])*m, 2, 3*m)
        self.assertTrue(np.all(exp == res))

        with self.assertRaises(DimensionError):
            np.insert(np.array([1, 2, 4])*m, 2, 3*s)

    def test_np_where(self):
        arr = np.arange(10)
        res = np.where(arr*m > 4*m, 1*m, 10*m)
        exp = np.where(arr > 4, 1, 10)*m
        self.assertTrue(np.all(res == exp))

    def test_np_histogram2d(self):
        a = np.arange(100)
        b = np.arange(100)*m
        hist, abins, bbins = np.histogram2d(a, b)

        exp_hist = np.histogram2d(np.arange(100), np.arange(100))[0]
        exp_abins = np.histogram2d(np.arange(100), np.arange(100))[1]
        exp_bbins = np.histogram2d(np.arange(100), np.arange(100))[2]*m
        self.assertTrue(np.all(hist == exp_hist))
        self.assertTrue(np.all(abins == exp_abins))
        self.assertTrue(np.all(bbins == exp_bbins))

    def test_scipy_integrate_solveivp(self):
        # Expected
        import scipy.integrate
        import numpy as np

        # in Ohms
        R = 10000
        # in Farad
        capa = 1*10**-12
        # time constant
        tau = R*capa
        # Source in volts
        Ve = 1
        # initial tension in volts
        y0 = [0]

        # def analytical_solution(t):
        #    return (y0[0]-Ve)*np.exp(-t/tau) + Ve

        def RHS_dydt(t, y):
            return 1/(tau)*(Ve - y)

        t_span = (0, 10*tau)

        # solution with no units
        solution_exp = scipy.integrate.solve_ivp(
            RHS_dydt,
            t_span,
            y0,
            dense_output=True,
        )

        from physipy import units, s, set_favunit

        ohm = units["ohm"]
        F = units["F"]
        V = units["V"]

        # in Ohms
        R = 10000 * ohm
        # in Farad
        capa = 1*10**-12 * F
        # time constant
        tau = R*capa
        # Source in volts
        Ve = 1 * V
        # initial tension in volts
        y0 = [0*V]

        def source_tension(t):
            return Ve

        def RHS_dydt(t, y):
            return 1/(tau)*(Ve - y)

        t_span = (0*s, 10*tau)
        solution_res = solve_ivp(
            RHS_dydt,
            t_span,
            y0,
            dense_output=True,
        )

        # things to test :
        # sol.t is in time dimension
        # sol.y has y dimension
        # sol.sol(time) returns a y dimension
        # sol.sol(dimless) raises
        self.assertTrue(np.all(solution_res.t == solution_exp.t*s))
        self.assertTrue(np.all(solution_res.y == solution_exp.y*V))
        ech_t = np.linspace(0*s, 2*tau)
        self.assertTrue(np.all(solution_res.sol(ech_t) ==
                        solution_exp.sol(ech_t.value)*V))
        with self.assertRaises(DimensionError):
            solution_res.sol(1)


if __name__ == "__main__":
    unittest.main()
