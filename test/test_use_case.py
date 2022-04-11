import numpy as np
import unittest
from fractions import Fraction
import math
import time

import pandas as pd
import matplotlib
import matplotlib.pyplot


from physipy.quantity import Dimension, Quantity, DimensionError
#from quantity import DISPLAY_DIGITS, EXP_THRESHOLD
from physipy.integrate import quad, dblquad, tplquad
from physipy.optimize import root, brentq
from physipy.quantity import vectorize #turn_scalar_to_str
from physipy.quantity.calculus import xvectorize, ndvectorize
from physipy.quantity import SI_units, units#, custom_units
from physipy.quantity import m, s, kg, A, cd, K, mol
from physipy.quantity import quantify, make_quantity, dimensionify
from physipy.quantity import check_dimension, set_favunit, dimension_and_favunit, drop_dimension, add_back_unit_param, decorate_with_various_unit, array_to_Q_array
from physipy.quantity.utils import asqarray
from physipy import imperial_units, setup_matplotlib
from physipy.quantity.utils import qarange


km = units["km"]
m = units["m"]
sr = units["sr"]


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
        cls.times = []
        cls.ids = []

    @classmethod
    def tearDownClass(cls):
        cls.df = pd.DataFrame.from_dict({
            "time":cls.times,
            "id":cls.ids,
        })
        
    def use_case_1(self):
        pass