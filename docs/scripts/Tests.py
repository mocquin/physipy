# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests

# %% [markdown]
# Tests are available in `\test` folder.

# %%
import sys
sys.path.insert(0, r"/Users/mocquin/MYLIB10/MODULES/physipy")

# %%
from test.test_dimension import TestClassDimension
from test.test_quantity import TestQuantity
from unittest.suite import TestSuite

# %%

# %%
# create the suite from the test classes
suite = TestSuite()
# load the tests
tests = unittest.TestLoader()

# add the tests to the suite
suite.addTests(tests.loadTestsFromTestCase(TestClassDimension))
suite.addTests(tests.loadTestsFromTestCase(TestQuantity))

# run the suite
runner = unittest.TextTestRunner()
runner.run(suite)

# %%
# !snakeviz prunsum_file

# %%
# %timeit 

# %%
# %prun -D prunsum_file -s nfl runner.run(suite)

# %%
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestQuantity)
runner = unittest.TextTestRunner()
runner.run(suite())

# %%
suite = unittest.TestSuite((TestClassDimension(), TestQuantity()))

runner = unittest.TextTestRunner()
runner.run(suite())

# %%
pwd

# %%
import unittest
def suite():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestClassDimension)
    suite.addTest(TestQuantity)
    return suite



runner = unittest.TextTestRunner()
runner.run(suite())

# %%

# %%
