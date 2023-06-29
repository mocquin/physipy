# -*- coding: utf-8 -*-
import os
import pathlib
import re

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

here = os.path.abspath(os.path.dirname(__file__))

VERSIONFILE = "physipy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

README = (HERE / "README.md").read_text()

setup(
    name="physipy",
    version=verstr,
    description="Manipulate physical quantities in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mocquin/physipy",
    author="mocquin",
    author_email="mocquin@me.com",
    license="MIT",
    keywords='physics physical unit units dimension quantity quantities',
    packages=find_packages(exclude=("test", "benchmarks")),
    # add content of MANIFEST
    include_package_data=True,
    install_requires=["numpy",
                      "scipy",
                      "sympy",
                      "matplotlib",
                      ]
)
