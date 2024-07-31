# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import physipy
import numpy as np
from physipy import m, s, Quantity, Dimension

arr = np.linspace(0, 200)
sca = 5.14
pi = np.pi
arr_m = arr * m


ech_lmbda_mum = np.linspace(2, 15)


class BenchDimension:
    def setup(self):
        self.length = Dimension("L")
        self.mass = Dimension("M")
        self.time = Dimension("T")

    def time_dimension_mul(self):
        self.length * self.mass

    def time_dimension_div(self):
        self.length / self.mass

    def time_dimension_pow(self):
        self.length**2

    def time_dimension_eq(self):
        self.length == self.length


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
        q = self.arr * m

    def time_QuantityAdd(self):
        q = m + m

    def mem_unit(self):
        """
        Memory consumption of the meter
        """
        return m

    def time_scalar_op_add(self):
        m + m

    def time_scalar_op_sub(self):
        m - m

    def time_scalar_op_mul(self):
        m * m

    def time_scalar_op_div(self):
        m / m

    def time_scalar_op_truediv(self):
        m // m

    def time_scalar_op_pow(self):
        m**1

    def time_use_case(self):
        x = arr * m
        x2 = sca * s**2
        y = x * x2 / pi * np.sum(x**2) + 3 * m**3 * s**2

    def time_arr_scalar_op_add(self):
        arr_m + m

    def time_arr_scalar_op_sub(self):
        arr_m - m

    def time_arr_scalar_op_mul(self):
        arr_m * m

    def time_arr_scalar_op_div(self):
        arr_m / m

    def time_arr_scalar_op_truediv(self):
        arr_m // m

    def time_arr_scalar_op_pow(self):
        arr_m**1

    def time_use_case2(self):
        from physipy import units, constants, K

        mum = units["mum"]
        hp = constants["h"]
        c = constants["c"]
        kB = constants["k"]

        def plancks_law(lmbda, Tbb):
            return (
                2
                * hp
                * c**2
                / lmbda**5
                * 1
                / (np.exp(hp * c / (lmbda * kB * Tbb)) - 1)
            )

        lmbdas = np.linspace(2, 5) * mum
        Tbb = 300 * K
        integral = np.trapz(plancks_law(lmbdas, Tbb), x=lmbdas)
