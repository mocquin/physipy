---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%matplotlib ipympl
from physipy import m, s
from physipy.qwidgets.plot_ui import WrappedFunction1D
from physipy.quantity.utils import name_eq
```

```python
@name_eq("Myfunc")        
def func(x1, x2, x3):
    return x1*x2 + 3 * x3

wf = WrappedFunction1D(func, 0*s, 5*s, 
                       x2=(0*m, 5*m),
                       x3=(0*m*s, 5*m*s))

wf
```

```python
wf.add_integral(1*s, 4*s)
```

```python

```
