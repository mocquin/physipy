---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

WIP, idea to set a context to ease conversion (K to °C, wavelength to wavenumber)

```python

from collections import UserList

# THIRD-PARTY
import numpy as np
import warnings

class Equivalency(UserList):
    """
    A container for a units equivalency.

    Attributes
    ----------
    name: `str`
        The name of the equivalency.
    kwargs: `dict`
        Any positional or keyword arguments used to make the equivalency.
    """

    def __init__(self, equiv_list, name='', kwargs=None):
        self.data = equiv_list
        self.name = [name]
        self.kwargs = [kwargs] if kwargs is not None else [dict()]

    def __add__(self, other):
        if isinstance(other, Equivalency):
            new = super().__add__(other)
            new.name = self.name[:] + other.name
            new.kwargs = self.kwargs[:] + other.kwargs
            return new
        else:
            return self.data.__add__(other)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.name == other.name and
                self.kwargs == other.kwargs)

```

```python

```
