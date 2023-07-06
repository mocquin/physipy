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

# QuantityQtSlider


A Qt slider that handles quantities and units

```python
from physipy import units, s, m
from physipy.qwidgets.qt import QuantityQtSlider
from PyQt5.QtWidgets import QApplication, QMainWindow

ohm = units["ohm"]

app = QApplication([])
win = QMainWindow()
w = QuantityQtSlider(2*ohm, 10*ohm, value=5*ohm, descr="Resistor")
win.setCentralWidget(w)
win.show()

if __name__ == '__main__':
    app.exec_()
```

# ParamSet

```python
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from physipy.qwidgets.qt import QuantityQtSlider
from physipy import units
V = units["V"]
C = units["C"]

ohm = units["ohm"]
#C = ohm = V = 1
```


```python
model = {'C': {"min":2*C, "max":5*C, "value":3*C},
         'R': {"min":2*ohm, "max":5*ohm, "value":3*ohm},
         'E': {"min":2*V, "max":5*V, "value":3*V},
        }
print(model)
```

```python
from physipy.qwidgets.qt import ParamSet

w = ParamSet(model)
```

```python
app = QApplication([])
win = QMainWindow()
win.setCentralWidget(w)
win.show()
app.exec_()
```

```python

```
