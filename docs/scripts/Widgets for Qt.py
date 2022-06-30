# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # QuantityQtSlider

# %% [markdown]
# A Qt slider that handles quantities and units

# %%
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

# %% [markdown]
# # ParamSet

# %%
from PyQt5.QtWidgets import QApplication, QMainWindow
from physipy import units
from physipy.qwidgets.qt import ParamSet
V = units["V"]
C = units["C"]

ohm = units["ohm"]


# %%
model = {'C': {"min":2*C, "max":5*C, "value":3*C},
         'R': {"min":2*ohm, "max":5*ohm, "value":3*ohm},
         'E': {"min":2*V, "max":5*V, "value":3*V},
        }
w = ParamSet(model)

# %%
app = QApplication([])
win = QMainWindow()
win.setCentralWidget(w)
win.show()
app.exec_()

# %% [markdown]
# # Model

# %% [markdown]
# See : 
#  - https://docs.python.org/3/howto/descriptor.html#properties

# %%
from physipy import units, s, m, asqarray
from physipy.qwidgets.qt import QuantityQtSlider
from physipy.quantity.utils import cached_property_depends_on
from pprint import pprint
from pydag import flatten_dep_dict

V = units["V"]
ohm = units["ohm"]
F = units["F"]

import numpy as np

import time


class PropertyDescriptor():
    
    def __init__(self, func_with_deps):
        self.func_with_deps = func_with_deps
        self.deps = func_with_deps.deps # (name, deps)
        
    def __set_name__(self, owner, name):
        # self.R
        self.__name__ = name
        self.public_name = name
        # actually refers to self._R_w
        self.private_name = '_' + name + "_with_deps"
        
    def __get__(self, obj, objtype=None):
        value = self.func_with_deps(obj)
        return value
    
    #def __set__(self, obj, func_with_deps):
    #    setattr(obj, self.private_name, func_with_deps)

        
def register_deps(name, deps, is_curve=False, pencolor=None, **kwargs):
    def decorator(f):
        f.deps = (name, deps, is_curve, pencolor, kwargs)
        return f
    return decorator

import functools

from functools import lru_cache
from operator import attrgetter, methodcaller


def cached_on_deps(*args):
    attrs = attrgetter(*args)
    
    # final decorator
    def decorator(func):
        # decorated once
        _cache = lru_cache(maxsize=None)(lambda self, _: func(self))
        
        # final decorated func
        def _with_tracked(self):
            return _cache(self, attrs(self))
        _with_tracked.deps = func.deps
        _with_tracked.__name__ = func.__name__
        return _with_tracked # final decorated fund
    
    return decorator # final decorator

class RegisteringType(type):
    def __init__(cls, name, bases, attrs):
        cls.curves = {}
        cls.dependent = {}
        
        for key, val in attrs.items():
            all_deps = getattr(val, 'deps', None)
            if all_deps is not None:
                (name, deps, is_curve, pencolor, kwargs) = all_deps
                cls.dependent[val.__name__] = (name, deps)
                # if is a curve
                if is_curve:
                    cls.curves[val.__name__] = (name, deps, pencolor)
        RAW_DICT = {}
        for xy, (param, deps) in cls.dependent.items():
            RAW_DICT[param] = deps
        cls.RAW_DICT = RAW_DICT
        print(cls.RAW_DICT)
        ## REGISTER BASE PARAMS
        cls.BASE_LIST = []
        cls.BASE_MINMAX = {}
        for key, val in attrs.items():
            if type(val)==ParamDescriptor:
                cls.BASE_LIST.append(key)
                cls.BASE_MINMAX[key] = {"min":val.min, "max":val.max}
                
                
        ## FLATTEN DEPENDCY DICT
        cls.FLAT_DICT = flatten_dep_dict(cls.RAW_DICT, cls.BASE_LIST)


        print("Created class with")
        pprint(cls.BASE_LIST)
        pprint(cls.RAW_DICT)
        pprint(cls.FLAT_DICT)

        
class ModelMixin():
    @property
    def params(self):
        param_dict = {}
        for pname in self.BASE_LIST:
            param_dict[pname] = {"value":getattr(self, pname), **self.BASE_MINMAX[pname]}
        return param_dict


class ParamDescriptor():
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
    def __set_name__(self, owner, name):
        # self.R
        self.public_name = name
        # actually refers to self._R_w
        self.private_name = '_' + name + "_observable_proxy_descriptor"
        
    def __set__(self, obj, qvalue):
        
        # to add the attribute name to the list of base params
        # but this means the check will be done at each __set__ 
        # so it slow down computation
        #if self.public_name not in obj.params:
        #    obj.params.append(self.public_name)
        if self.min <= qvalue <= self.max:
            setattr(obj, self.private_name, qvalue)
        else:
            raise ValueError
    
    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        return value
    
    


class ModelRC(ModelMixin, metaclass=RegisteringType):
    
    R  = ParamDescriptor(0*ohm, 10*ohm)
    C  = ParamDescriptor(0*F, 10*F)
    Ve = ParamDescriptor(0*V, 10*V)
    u0 = ParamDescriptor(0*V, 10*V)

    def __init__(self, Ve, R, C, u0=0*V, ech_t=np.arange(100)*s):
        self.R = R
        self.C = C
        self.Ve = Ve
        self.u0 = u0
        self.ech_t = ech_t


    @cached_on_deps('R', 'C') # will not recompute if R and C state are unchanged
    @register_deps("tau", ["R", "C"])#, latex='\tau=R\cdotC')
    def tau(self):
        return self.R * self.C
    #tau = register_deps("tau", ["R", "C"])(tau)
    
    #@cached_on_deps('Ve')
    @cached_on_deps('Ve') # will not recompute if R and C state are unchange
    @register_deps("convergence", ["Ve"], True, pencolor="r")
    def xy_convergence(self):
        return asqarray([0*s, np.max(self.ech_t)]), asqarray([self.Ve, self.Ve])
    
    #@cached_property_depends_on('u0', 'Ve', "tau") # will not recompute if R and C state are unchanged
    #@cached_on_deps('Ve', 'u0', "tau") # will not recompute if R and C state are unchange
    @cached_on_deps('Ve', 'u0', 'R', 'C') # will not recompute if R and C state are unchange
    @register_deps("slope at start", ["u0", "Ve", "tau"], True, pencolor="g", latex='\frac{V_e-u_0}{\tau}')#"R", "C"])
    def xy_slope_at_start(self):
        xs = asqarray([0*s, self.tau(), self.tau()])
        ys = asqarray([self.u0, self.Ve, self.u0])
        return xs, ys
    
    @cached_on_deps('Ve', 'u0', 'R', 'C') # will not recompute if R and C state are unchange
    @register_deps("response", ["u0", "Ve", "tau"], True, pencolor="b", latex="(u_0-V_e)e^{-t/\tau}+V_e")#"R", "C"])
    def xy_response(self, ech_t=None):
        if ech_t is None:
            ech_t = self.ech_t
        xs = ech_t
        ys = (self.u0 - self.Ve) * np.exp(-ech_t/self.tau()) + self.Ve
        return xs, ys 

# %%
model = ModelRC(0*V, 0*ohm, 3*F)
pprint(model.params)
print(model.R, model.tau())
model.R = 3*ohm
pprint(model.R)
pprint(model.tau())
pprint(model.params)
print(model.xy_convergence())


# %%
print(model.tau())
pprint(model.curves)
pprint(model.params)

# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QApplication, QVBoxLayout, QLabel, QMainWindow
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QLineEdit, QComboBox
import PyQt5.QtWidgets
import PyQt5.QtCore as QtCore

import pyqtgraph.dockarea


import pyqtgraph
import pyqtgraph as pg

pyqtgraph.setConfigOption('background', '#0f4c5c')
pyqtgraph.setConfigOption('foreground', 'w')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)
import pyqtgraph.console

class VuePyQt(QMainWindow):#QWidget):
    def __init__(self, model, parent = None):
        super(VuePyQt, self).__init__()

        self.model = model
        self.setStyleSheet("background-color: #0f4c5c;")

        area = pyqtgraph.dockarea.DockArea()
        d1 = pyqtgraph.dockarea.Dock("Parameters", size=(1, 1))    
        d2 = pyqtgraph.dockarea.Dock("Plot", size=(500,300), closable=False)
        area.addDock(d1, 'top') 
        area.addDock(d2, 'bottom', d1)   
        
        d1.label.setStyleSheet("background-color:#0f4c5c;") # color falls back when moved
        d2.label.setStyleSheet("background-color:#0f4c5c;") # color falls back when moved
        
    
        # create a Vertical layout
        layout = QVBoxLayout()
        self.setContentsMargins(0, 0, 0, 0) 
        #self.setSpacing(0)
        
        # spent 3 days on this : https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
        widgets = []
        for key, value in model.params.items():
            #print(key, value)
            # create slider
            if hasattr(value["min"], "value"):
                slider = QuantityQtSlider(value["min"], 
                                          value["max"],
                                          value=value["value"], descr=key)
            else:
                slider = QuantityQtSlider(quantify(value["min"]), 
                                          quantify(value["max"]),
                                          value=quantify(value["value"]), descr=key)

            # add slider to Vue
            setattr(self, key+"_slider", slider)
            # connect slider's value to model's value
            getattr(self, key+"_slider").qtslider.valueChanged.connect(lambda qtvalue, key=key:self.set_attr(self.model, key))#(lambda qtvalue:self.update_model_param_value(qtvalue, slider, key))
            # make slider to update all curves
            #getattr(self, key+"_slider").qtslider.valueChanged.connect(lambda qtvalue:self.update_traces(qtvalue))
            
            
            #print("make slider update dependent trace for", key)
            #print("  looping in curves")
            # make slider to update dependent traces
            for k, v in self.model.curves.items():
                #print("     - curve ", k, v)
                base_deps = self.model.FLAT_DICT[v[0]]
                #print("        with deps", base_deps)
                if key in base_deps:
                    #print("        key", key, "is in deps")
                    #if key in v[1]: # loop over parameter list
                    func = getattr(self.model, k)
                    # func=func and k=k are mandatory see SO's 2295290
                    def _upd(qt_value, func=func, k=k):
                        xs, ys = func()
                        self.traces[k].setData(xs.value,ys.value)
                    getattr(self, key+"_slider").qtslider.valueChanged.connect(_upd)
                    #print("-------final setting", key)
            
            
            

            widgets.append(slider)
            
        # add widgets to the V layout, top to bottom
        for w in widgets:
            layout.addWidget(w)

#        layout.setAlignment(QtCore.Qt.AlignTop)
        
        # to remove margins 
        layout.setContentsMargins(0, 0, 0, 0) #left top right bot
        # to remove space between each HBox
        layout.setSpacing(0)
        
        self.win = pg.GraphicsWindow(title="Basic plotting examples")
        #self.win.setStyleSheet("background-color: background-color: #6d6875;")

        self.canvas = self.win.addPlot(title="Plot11", row=1, col=1,)# axisItems={"bottom":sp_xaxis})
        self.canvas.setLabel('left', 'Y-axis Values')
        self.canvas.addLegend()
        self.canvas.showGrid(x=True, y=True)
        #self.canvas.se ("#6d6875")
        #self.canvas.setStyleSheet("background-color: background-color: #6d6875;")

        
        self.traces = dict()
        
        for name, all_deps in self.model.curves.items():
            func_name = name#trace_dict["xys"]
            xs, ys = getattr(self.model, func_name)()
            pen_color = all_deps[-1]
            self.trace(name, xs, ys, pen=pen_color)
        
        # add the GraphicsWindow widget to the V layout
        #layout.addWidget(self.win)

        # since using a QMainWindow, we use a widget to set as central widget
        # and set the layout of this widget
        widget = QWidget()
        widget.setLayout(layout)
        
        d1.addWidget(widget)
        d2.addWidget(self.win)
        self.setCentralWidget(area)
        #self.setCentralWidget(widget)

    def set_attr(self, obj, key):
        setattr(obj, key, getattr(self, key+"_slider").value)

    def update_model_param_value(self, qtvalue, slider, key):
        setattr(self.model, key, slider.value)
    
    def trace(self, name, dataset_x,dataset_y, pen="y", symbol="o"):
        if name in self.traces:
            self.traces[name].setData(dataset_x.value,dataset_y.value)
        else:
            self.traces[name] = self.canvas.plot(x=dataset_x.value, y=dataset_y.value, pen=pen, width=3, symbol=symbol, name=name, symbolBrush=pen)
            #pen =(0, 0, 200), symbolBrush =(0, 0, 200),
                      #symbolPen ='w', symbol ='o', symbolSize = 14, name ="symbol ='o'")
    
    #def update_trace_generator(self, name, curve_func):
    #    def update_func():
    #        xs, ys = curve_func()
    #        self.trace(name, xs, ys)
    #    return update_func
    
    #def update_traces(self, qtvalue):
    #    for name, trace_dict in self.model.curves.items():
    #        func = trace_dict["xys"]
    #        xs, ys = func()
    #        pen_color = trace_dict["pen_color"]
    #        self.trace(name, xs, ys, pen=pen_color)


# %%
        
def main():
    app = QApplication(sys.argv)
    
    # to define a custom color
    #from PyQt5.QtGui import QPalette
    #palette = QPalette()
    #palette.setColor(QPalette.ButtonText, Qt.red)
    #app.setPalette(palette)
    
    Ve = 5 * V
    R  = 3 * ohm
    C  = 2 * F
    
    model = ModelRC(Ve, R, C)
    vue = VuePyQt(model)
    print("vue's traces : ", vue.traces)
    vue.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

# %%
