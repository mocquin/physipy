# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import physipy
from physipy import m
import numpy as np

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    #def time_keys(self):
    #    for key in self.d.keys():
    #        pass

    #def time_iterkeys(self):
    #    for key in self.d.iterkeys():
    #        pass

    def time_range(self):
        d = self.d
        for key in range(500):
            x = d[key]

    #def time_xrange(self):
    #    d = self.d
    #    for key in xrange(500):
    #        x = d[key]


class BasicPhysipy:
    def setup(self):
        self.arr = np.arange(10)
        
        
    def time_QuantityCreation(self):
        q = physipy.Quantity(1, physipy.Dimension("M"))
        
    def time_QuantityCreationByMul(self):
        q = 2 * m

    def time_QuantityCreationByExpr(self):
        q = physipy.Quantity(1, physipy.Dimension("M/L"))
        
    def time_QuantityCreationByArray(self):
        q = np.arange(10) * m
        
    def time_QuantityAdd(self):
        q = m + m
        
    def mem_unit(self):
        """
        Memory consumption of the meter
        """
        return m