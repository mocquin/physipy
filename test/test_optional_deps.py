"""
Guard the optional-dependency contract : `import physipy` and core
dimensional arithmetic must work without importing scipy or matplotlib, which
are only needed for physipy.calculus / physipy.constants (scipy) and the
plotting integration (matplotlib).

The checks run in a fresh subprocess interpreter because other tests in the
suite import scipy / matplotlib into the shared process.
"""

import subprocess
import sys
import textwrap
import unittest


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


class TestLazyOptionalDeps(unittest.TestCase):
    def test_import_physipy_does_not_load_scipy_or_matplotlib(self):
        res = _run(
            """
            import sys
            import physipy
            from physipy import m, s
            _ = (2 * m) / (1 * s)          # core arithmetic
            assert "scipy" not in sys.modules, "scipy eagerly imported"
            assert "matplotlib" not in sys.modules, "matplotlib eagerly imported"
            print("OK")
            """
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK", res.stdout)

    def test_constants_access_loads_scipy(self):
        res = _run(
            """
            import sys
            import physipy
            assert "scipy" not in sys.modules
            _ = physipy.constants["c"]
            assert "scipy" in sys.modules
            print("OK")
            """
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK", res.stdout)

    def test_plot_helpers_load_matplotlib(self):
        res = _run(
            """
            import sys
            import physipy
            assert "matplotlib" not in sys.modules
            from physipy import setup_matplotlib
            assert "matplotlib" in sys.modules
            print("OK")
            """
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
