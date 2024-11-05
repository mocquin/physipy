# Setup the environment

## General checks

 - `perflint physipy`

## Release

### On Github
Once the package is shipped to pypi, you can create a release on Github : ([help from github](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)) : 
 - Go to the package home
 - On the right-side menu, click on Release
 - Draft a new release
 - Select the tag version
 - Write a title and description
 - The source code from the repo are packed in zip/tar.gz files. You can add wheel or other packages by drag-n-drop.


## Testing

### Type hints, type checking
`mypy ./physipy`

### With doctest
Run each file with doctest: 
```
python -m doctest -v physipy/quantity/dimension.py
```

Another approach is to use pytest to run all tests, including the doctests 
```
pytest --doctest-modules
pytest -v --doctest-modules --doctest-glob="*.py"
pytest -vv --doctest-modules --doctest-glob="*.py" .\physipy\
```
Note that using `python -m doctest -v .\physipy\quantity\utils.py` (for example) fails because of relative import, as the script is considered a stand-alone file.

## Sorting imports
A simple good practice is to sort imports using the isort package:
`pip install isort`
`isort .\physipy\ --verbose`
To run just check to see if anything import should be sorted : 
`isort . --check-only`

## Formatting
pycodestyle --config=setup.cfg --verbose --statistics -qq .
Given an error number : 
pycodestyle --show-source --select=E722 .

## Benchmarking versions

Benchmark results using [asv](https://github.com/airspeed-velocity/asv) are available at [https://mocquin.github.io/physipy/](https://mocquin.github.io/physipy/) :

[![./../ressources/asv_screenshot.png](./../ressources/asv_screenshot.png)](https://mocquin.github.io/physipy/)

For more information, see the dedicated documentation page on [benchmarking with airspeedvelocity].

## For documentation

The documentation system of physipy is based on [mkdocs]. It relies additionnaly on [mkdocs-material] for easy customisation and [mkdocstrings] to automaticaly include all physipy's docstrings into the documentation.

 - `pip install mkdocs`
 - `pip install mkdocs-material`
 - `pip install mkdocstrings-python`
 - `pip install markdown_include` : to include the repo README.md directly into the documentation site

The documentation configuration is set in the `mkdocs.yml` configuration file at the root of the project.

The raw documentation is stored in the [docs] directory as markdown files. Markdown files can converted back-and-forth with notebook format, using : 
```
jupyter nbconvert --to markdown .\docs\scientific-stack\math-support.ipynb # from nb to md
jupytext --to ipynb .\docs\scientific-stack\numpy-support.md               # from md to nb
```

During development, cd to physipy root directory, and run the following command to launch a live server :
`mkdocs serve`

Please note that this only serves the documentation, and does not export it to an actual static-website.
To export the documentation to a static-website use : 
`mkdocs build`

This will generate the content to a directory given by `site_dir` in the `mkdocs.yml` configuration file (currently in `_mkdocs_site`).

To pin the packages needed for the doc, create a new venv, install pip-tools, then the packages specified in docs/requirements.in. Use `mkdocs serve` to review the docs, and then `pip-compile docs/requirements.in` to create a `docs/requirements.txt` file.