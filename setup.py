# -*- coding: utf-8 -*-
import os
import pathlib
import re

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

here = os.path.abspath(os.path.dirname(__file__))


### reading version from _version.py file
VERSIONFILE="physipy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="physipy",
    version=find_version("physipy", "__init__.py"),
    description="Manipulate physical quantities in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mocquin/physipy",
    author="mocquin",
    author_email="mocquin@me.com",
    license="MIT",
    keywords='physics physical unit units dimension quantity quantities',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests")), #["physipy"], a list of all Python import packages that should be included in the distribution package
    include_package_data=True,
    install_requires=["scipy", "sympy", "numpy"],
#    entry_points={
#        "console_scripts": [
#            "realpython=reader.__main__:main",
#        ]
#    },
)

#if __name__ == "__main__":
#    print(verstr)
