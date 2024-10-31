import doctest
import unittest

from physipy import calculus
from physipy import math as physipy_math
from physipy.quantity import dimension, quantity, utils


# The load_tests() function is automatically called by unittest
# and the returned 'tests' are added
# see https://docs.python.org/3/library/doctest.html#unittest-api
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(dimension))
    # /!\ dimension doctest is tested in test_dimension
    tests.addTests(doctest.DocTestSuite(quantity))
    tests.addTests(doctest.DocTestSuite(calculus))
    # TODO : dict and module share the same name
    # tests.addTests(doctest.DocTestSuite(physipy.quantity.units))
    tests.addTests(doctest.DocTestSuite(utils))
    # TODO : dict and module share the same name
    # tests.addTests(doctest.DocTestSuite(constants))
    tests.addTests(doctest.DocTestSuite(physipy_math))
    return tests


if __name__ == "__main__":
    # Run the tests with verbosity level 2 for detailed output
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(load_tests(unittest.TestLoader(), unittest.TestSuite(), None))
