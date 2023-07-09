import unittest
import pycodestyle
import time

class TestCodeFormat(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        self.tottime = 0

    def tearDown(self):
        t = time.time() - self.startTime
        self.tottime = self.tottime + t
        print(f"{self.id():70} : {t:10.6f}")
        

    # TODO : improve style and uncomment this test
    #def test_conformance(self):
    #    """Test that we conform to PEP-8."""
    #    style = pycodestyle.StyleGuide(quiet=True)
    #    result = style.check_files('*.py')
    #    self.assertEqual(result.total_errors, 0,
    #                     "Found code style errors (and warnings).")

if __name__ == "__main__":
    unittest.main()
   
