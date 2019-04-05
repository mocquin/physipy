# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="physipy",
    version="0.1.0",
    description="Manipulate physical quantities in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mocquin/physipy",
    author="mocquin",
    author_email="mocquin@me.com",
    license="MIT",
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
