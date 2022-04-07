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

 - https://en.wikipedia.org/wiki/Observer_pattern#Python
 - https://en.wikipedia.org/wiki/Reactive_programming
 - https://stackoverflow.com/questions/6190468/how-to-trigger-function-on-value-change
 - https://www.python.org/doc/essays/graphs/ 
 - https://stackoverflow.com/questions/48336820/using-decorators-to-implement-observer-pattern-in-python3
 - https://github.com/victorcmoura/notifyr/tree/master/notifyr
 - https://codereview.stackexchange.com/questions/253675/a-python-decorator-for-an-observable-property-that-notifies-observers-when-th/253714#253714?newreg=04c9ad06b7164ef9b8fc4a7f32af2ee9


```python

```

```python

```

# Using numpy/broadcasting : defining a set of inputs for a model

```python
from physipy import m, s, K, kg
import numpy as np
```

Creation of combinations is no "simple" but very doable using meshgrid : but be carefull, combinations grow **FAST** : 

```python
# uniform sampling
lenghts = np.linspace(0.4, 0.8, num=30)*m
times = np.linspace(1, 10, num=11)*s
temps = np.linspace(300, 310, num=11)*K
# random sampling
mass = np.random.randn(100)*3*kg + 100*kg


LENGTHS, TIMES, TEMPS, MASS= np.meshgrid(lenghts, times, temps, mass)

lenghts = LENGTHS.flatten()
times = TIMES.flatten()
temps = TEMPS.flatten()
mass = MASS.flatten()
```

```python
print(mass.shape)
```

Once again, thanks to numpy, we can make a computation for all possibilities at once : 

```python
some_metric = lenghts*times/temps**2*np.exp(mass/(50*kg))
```

```python
some_metric
```

# Cache a property with dependency


Based on https://stackoverflow.com/questions/48262273/python-bookkeeping-dependencies-in-cached-attributes-that-might-change


Let's make a simply RC model :


## Using a builtin decorator `cached_property_depends_on`


PROS : 
 - only a decorator needed
CONS : 
 - no triggering at value change or dependcy graph

```python
import time
from functools import lru_cache
from operator import attrgetter

from physipy.quantity.utils import cached_property_depends_on

class BADTimeConstantRC:
    
    def __init__(self, R, C):
        self.R = R
        self.C = np.linspace(0*Farad, C)
        
    @property
    def tau(self):
        print("Slow computation...")
        time.sleep(5)
        return self.R * self.C
    
class GOODTimeConstantRC:
    
    def __init__(self, R, C):
        self.R = R
        self.C = np.linspace(0*Farad, C)
    
    @cached_property_depends_on('R', 'C')
    def tau(self):
        print("Slow computation...")
        time.sleep(5)
        return self.R * self.C
    
    
from physipy import units
ohm = units["ohm"]
Farad = units["F"]
bad = BADTimeConstantRC(ohm, Farad)
print("Bad first : ", bad.tau) # This is long the first time...
print("Bad second : ", bad.tau) # ... but also the second time !

good = GOODTimeConstantRC(ohm, Farad)
print("Good fisrt : ", good.tau) # This is long the first time...
print("Good second : ", good.tau) # ... but not the second time since neither R nor C have changed.
```

## using the acylic_model


This is another way to write a model with dependencies. Like the previous one, changing a parameter doesn't immediately triggger a computation of dependendant parameters, only at get-time.


PROS : 
 - dependency graph automatic  

CONS : 
 - definition of parameter is disinguished from computation method

```python
from physipy import units, s
F = units["F"]
ohm = units["ohm"]

from physipy.quantity._acyclic_model import IndependentAttr, DeterminantAttr

class TimeConstant():
    # This class corrects the dependency problems in the DataflowFail class by using the following descriptors:
    # The following defines the directed acyclic computation graph for these attributes.
    R = IndependentAttr(ohm, 'R')
    C = IndependentAttr(F,  'C')
    tau = DeterminantAttr(['R', "C"], '_tau', 'tau')

    def _tau(self):
        print("Throu _tau")
        self.tau = self.R * self.C


tc = TimeConstant()
print(tc.R, tc.C)
print(tc.tau)
tc.R = 2*ohm
print(tc.tau)
print(tc.tau)
tc.R = 2*ohm
print(tc.tau)
```

<!-- #region tags=[] -->
# Better increase the number of MC samples rather than cross-product them
<!-- #endregion -->

Say you want to simulate random distributions of 2 variables : should you use 100 samples for one, and 100 for the other and then compute all the 100x100=10000 couples possibles OR use 10000 samples for one and 10000 random samples for the other, and just use the 10000 couples.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 100
x1 = np.random.randn(N)
x2 = np.random.randn(N)

bins=20
xmin=ymin=-3
xmax=ymax=3
fig, axes = plt.subplots(1,3, sharex=True, sharey=True, figsize=(16,8))


axes[0].hist2d(x1, x2, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
axes[0].scatter(x1, x2, alpha=50/len(x1), facecolors='none', edgecolors="r")
sns.kdeplot(x1, x2, ax=axes[0])

X1, X2 = np.meshgrid(x1, x2)

axes[1].hist2d(X1.flatten(), X2.flatten(), bins=bins, range=[[xmin, xmax], [ymin, ymax]])
axes[1].scatter(X1.flatten(), X2.flatten(), alpha=1000/len(X1.flatten()), facecolors='none', edgecolors="r")
sns.kdeplot(X1.flatten(), X2.flatten(), ax=axes[1])

x1_bis = np.random.randn(X1.flatten().size)
x2_bis = np.random.randn(X1.flatten().size)

axes[2].hist2d(x1_bis, x2_bis, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
axes[2].scatter(x1_bis, x2_bis, alpha=1000/len(x1_bis), facecolors='none', edgecolors="r")
sns.kdeplot(x1_bis, x2_bis, ax=axes[2])
```

# Introspection


Simply use pandas and seaborn, since the inputs and ouputs are basically vectors of data :

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
df = pd.DataFrame({
    "length":lenghts.value,
    "times":times.value,
    "temps":temps.value,
    "mass":mass.value,
    "metric":some_metric.value,
})
```

```python
sns.pairplot(df, hue="times")
plt.figure()
sns.heatmap(df.corr(), annot=True)
```

```python

```

<!-- #region tags=[] -->
# Dynamic models
<!-- #endregion -->

## Using traitlets and widgets


Lets try to make the RC model using traitlets


PROS : 
 - 
 
CONS :  
 - tau and compute_tau are separeted

```python
import traitlets
import ipywidgets as ipyw
from physipy import units, s, Quantity
from physipy.qwidgets.qipywidgets import QuantityTextSlider, QuantityText, QuantitySliderDescriptor
F = units["F"]
ohm = units["ohm"]

    
class QuantityLabelDescriptor():
    
    def __init__(self, deps_names, cb_name):
        self.deps_names = deps_names
        self.cb_name = cb_name
        
    
    def __set_name__(self, owner, name):
        print("setting name for label", name)
        # self.R
        self.public_name = name
        # actually refers to self._R_w
        self.private_name_w=  name + "_w"
        # 
        self.private_name_q = name + "_q"
        
        ## if widget already exists
        #if hasattr(owner, self.private_name):
        #    # get the private widget value, and set its value to the result of compute_
        #    setattr(getattr(owner, self.private_name), "value", getattr(owner, "compute_"+self.public_name)())
        #else:
        #    print("create new widget")
        #    setattr(owner, self.private_name, ipyw.Label(self.public_name + ":" + ""))
        #    # get the widget
        #    w = getattr(owner, self.private_name)
        #    # set the widget value to
        #    setattr(w, "value", str(getattr(owner, "compute_"+self.public_name)(owner)))

            
    def _update_text(self, obj):
        setattr(obj, self.private_name_w+"value", str(getattr(obj, self.private_name_q)))
    
    
    def __get__(self, obj, objtype):
        # quantity never computed yet
        for dep in self.deps_names:
            # get the ref to the widget of the dependency
            print('setting the callbacks')
            getattr(obj, dep+"_w").observe(lambda e:self._update_text(obj), "value")
        
        if hasattr(obj, self.private_name_q):
            return getattr(obj, self.private_name_q)
        else:

            print("trying to compute")
            res = getattr(obj, "compute_"+self.public_name)()
            setattr(obj, self.private_name_q, res)
            setattr(obj, self.private_name_w, ipyw.Label(self.public_name + ":" + str(res)))
            
            return res
    #def __get__(self, obj, objtype):
    #    if hasattr(obj, self.private_name):
    #        setattr(getattr(obj, self.private_name), "value", str(getattr(obj, "compute_"+self.public_name)()))
    #    else:
    #        print("create new widget")
    #        setattr(obj, self.private_name, ipyw.Label(self.public_name + ":" + ""))
    #    for dep in self.deps_names:
    #        # get the ref to the widget of the dependency
    #        print('setting the callbacks')
    #        getattr(obj, "_"+dep+"_w").observe(self._update_text(obj), "value")
    #    if not hasattr(obj, self.private_name):
    #        res = getattr(obj, "compute_"+self.public_name)()
    #        setattr(obj, self.private_name, ipyw.Label(self.public_name + ":" + str(res)))
    #        return res
#
    #    return getattr(obj, "compute_"+self.public_name)()
    
    def __set__(self, obj, value):
        if hasattr(obj, self.private_name_w):
            # set the actual quantity value
            setattr(obj, self.private_name_q, value)
            self._update_text(obj)
        else:
            print("create new widget")
            setattr(obj, self.private_name, ipyw.Label(self.public_name + ":" + str(value)))
            


        
class TraitletsRC():
    """
    To introspect descriptors : 
        # introspect the descriptors/attributes
        print(vars(TraitletsRC(2*ohm, 2*F)))
        # introspect the slider
        print(vars(vars(TraitletsRC(2*ohm, 2*F))["_R_w"]))
    """
    
    R = QuantitySliderDescriptor()
    C = QuantitySliderDescriptor(min=0.01*F, max=200*F)
    tau = QuantityLabelDescriptor(["R", "C"], "compute_tau")

    def __init__(self, R, C):
        self.R = R
        self.C = C
        
    
    def compute_tau(self):
        return self.R * self.C
    
    def __repr__(self):
        display(self.R_w)
        display(self.C_w)
        display(self.tau_w)
        return ""
    
rc = TraitletsRC(2*ohm, 2*F)
print(vars(rc).keys())
print(rc.R, rc.C)
print(rc.tau)
#print(rc.tau)
display(rc.R)
display(rc.C)
display(rc.C_w)
rc
```

```python
rc.R = 4*ohm
```

```python
rc.tau
```

```python
for i in vars(rc):
    print(i)
```

<!-- #region tags=[] -->
## Using acyclic_model : dependent model
Using the acyclic module
<!-- #endregion -->

```python
from physipy import units, set_favunit, s
from physipy.quantity._acyclic_model import IndependentAttr, DeterminantAttr
import numpy as np
from numpy import exp
ms = units["ms"]
V = units["V"]
F = units["F"]
ohm = units["ohm"]


class IRC2():
    R  = IndependentAttr(1*ohm, 'R')
    C  = IndependentAttr(1*F, 'C')
    Ve = IndependentAttr(1*V, 'Ve')
    u0 = IndependentAttr(0*V, 'u0')
    
    tau = DeterminantAttr(['R', 'C'], 'compute_tau', 'tau')
    u = DeterminantAttr(['R', 'C', "Ve", "u0"], 'compute_u', 'u')

    
    def compute_tau(self):
        self.tau = self.R * self.C
        
    def compute_u(self):
        self.u = (self.u0 - self.Ve)*exp(-3*s/self.tau) + self.Ve
```

<!-- #region tags=[] -->
## ObservableQuanttity class: observable pattern
 - Standalone Quantity-Like class for a quantity
 - Update a quantity's ".value" will trigger callbacks.
<!-- #endregion -->

```python
from physipy import units
from physipy import Quantity

class ObservableQuantity(Quantity):
    
    def __init__(self, *args, cb_for_init=None, **kwargs):
        self._callbacks = []
        if cb_for_init is not None:
            self._callbacks.append(cb_for_init)
        super().__init__(*args, **kwargs)
    
    def register_callback_when_value_changes(self, cb):
        """
        Register a callback that will be called on value change.
        """
        self._callbacks.append(cb)
    
    def notify_change_in_value(self, change):
        """
        Activate all callbacks that were registered.
        """
        for callback in self._callbacks:
            callback(change)
    
    @Quantity.value.setter
    def value(self, value):
        """
        Based on Quantity.value.setter
        """
        # handle the initial setting value, we can't "get" it yet
        try:
            old = self.value
        except:
            old = "NotDefinedYet"
        if isinstance(value, (list, tuple)):
            self._value = np.array(value)
        else:
            self._value = value
        change = {"old":old, "new":value}
        self.notify_change_in_value(change)    

obs_r = ObservableQuantity(2, ohm.dimension, cb_for_init=lambda _:print("value changed"))
print(obs_r)
print(obs_r*2)
print(obs_r**2 + obs_r**2)

obs_r.register_callback_when_value_changes(lambda change:print("valuechanged with change", change))
obs_r.value = 1
```

```python
print(obs_r)
print(obs_r*2, type(obs_r*2))
print(obs_r+obs_r, type(obs_r+obs_r))
```

## Using a descriptor approach


 - Registering a Quantity by declaring the other quantities that should be updated.
 - Dependent tau is immediately computed and updated.
 - For class model that wraps several quantities

```python
from physipy import units, s, set_favunit, setup_matplotlib
setup_matplotlib()
import numpy as np
import matplotlib.pyplot as plt
from physipy.quantity._observables_model import ObservableQuantityDescriptor

ms = units["ms"]
ohm = units["ohm"]
F = units["F"]
V = units["V"]

class RC():
    
    R = ObservableQuantityDescriptor(["tau"])
    C = ObservableQuantityDescriptor(["tau"])
    tau = ObservableQuantityDescriptor()
    
    def __init__(self, R, C):
        self.R = R
        self.C = C
    
    def compute_tau(self, change):
        print("updating tau with", change)
        self.tau = self.R * self.C
        self.tau.favunit = ms

    @property
    def _state(self):
        return self._observables_dict
        
rc = RC(1*ohm, 1*F)
print("First getting tau")
print(rc.C, rc.R, rc.tau)

print("Now changing R")
rc.R = 2*ohm
print("Second getting tau")
print(rc.C, rc.R, rc.tau)

print(rc._state)
```

```python
%matplotlib ipympl

class RC_plot():
    
    R = ObservableQuantityDescriptor(["tau"])
    C = ObservableQuantityDescriptor(["tau"])
    tau = ObservableQuantityDescriptor(["response_val"])
    response_val = ObservableQuantityDescriptor()
    
    def __init__(self, R, C, ech_t):
        self.R = R
        self.C = C
        self.ech_t = ech_t
    
    def compute_tau(self, change):
        print("updating tau with", change)
        self.tau = self.R * self.C
        self.tau.favunit = ms
        
    def compute_response_val(self, change):
        print("updating response_line with", change)
        self.response_val = self.response(self.ech_t)
        try:
            self.resp_line.set_ydata(self.response_val)
        except:
            pass
        
    @set_favunit(V)
    def response(self, t):
        return 1 - np.exp(-t/self.tau)

    def plot(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        
        resp_line = self.ax.plot(self.ech_t, self.response_val)[0]
        self.resp_line = resp_line

    def get_descriptors(self):
        for k, v in vars(rc).items():
            if k.endswith("descriptor"):
                print(k, v)
        
rc = RC_plot(1*ohm, 1*F, np.arange(10, step=0.01)*s)
rc.plot()
```

```python
rc.R
```

```python
rc.R = rc.R /2
```

```python
for k, v in rc._observables_dict.items():
    print(k, type(v))
```

<!-- #region tags=[] -->
## Using autocalc
<!-- #endregion -->

Not working out of the box for physipy, but could be done with a little work.

- https://medium.com/towards-data-science/create-a-simple-app-quickly-using-jupyter-notebook-312bdbb9d224
 - https://autocalc.readthedocs.io/en/latest/index.html
 - https://github.com/kefirbandi/autocalc


```python
import ipywidgets as widgets
from autocalc.autocalc import Var
import math

a = Var('a', initial_value = 1, widget = widgets.FloatText())
b = Var('b', initial_value = -3, widget = widgets.FloatText())
c = Var('c', initial_value = 1, widget = widgets.FloatText())

display(a); display(b); display(c)


def Dfun(a, b, c):
    try:
        return math.sqrt(b*b-4*a*c)
    except ValueError:
        return math.nan

def x1fun(a,b,D):
    return (-b-D)/2/a
def x2fun(a,b,D):
    return (-b+D)/2/a


D = Var('D', fun=Dfun, inputs=[a, b, c])
x1 = Var('X1', fun=x1fun, inputs=[a, b, D], widget = widgets.FloatText(), read_only=True)
x2 = Var('X2', fun=x2fun, inputs=[a, b, D], widget = widgets.FloatText(), read_only=True)
display(x1)
display(x2)
```

```python
a = Var('a', initial_value = 1*m, widget = QuantityText())
b = Var('b', initial_value = -3*m, widget = QuantityText())
c = Var('c', initial_value = 1*m, widget = QuantityText())

display(a); display(b); display(c)


def Dfun(a, b, c):
    try:
        res = (b*b-4*a*c)**0.5
        print(res)
        return res
    except ValueError:
        return math.nan

def x1fun(a,b,D):
    res = (-b-D)/2/a
    return res
def x2fun(a,b,D):
    return (-b+D)/2/a


D = Var('D', fun=Dfun, inputs=[a, b, c])
x1 = Var('X1', fun=x1fun, inputs=[a, b, D], widget = QuantityText(), read_only=True)
x2 = Var('X2', fun=x2fun, inputs=[a, b, D], widget = QuantityText(), read_only=True)
display(x1)
display(x2)
```

## pyqtgraph parameter tree


Not what I thought, basicaly just nested parameters


# Observable descriptor with dependency

```python
from physipy import units

W = units["W"]


class ObservableProperty:
    
    def __set_name__(self, owner: type, name: str) -> None:
        
        # generic callback setter to instance
        def add_observer(obj, observer):
            if not hasattr(obj, self.observers_name):
                setattr(obj, self.observers_name, [])
            getattr(obj, self.observers_name).append(observer)
            
        self.private_name = f'_{name}'
        self.observers_name = f'_{name}_observers'
        
        # add the callback setter to the instance
        setattr(owner, f'add_{name}_observer', add_observer)
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)
        for observer in getattr(obj, self.observers_name, []):
            # using obj here allows to use "self.toto" in callbacks
            observer(obj, value)
         
        

class Car():
    
    wheels = ObservableProperty()
    power = ObservableProperty()
    
    def __init__(self, wheels, power):
        self.wheels = wheels
        self.power = power
        
    
    @property
    def power_per_wheel(self):
        return self.power/self.wheels

    def __repr__(self):
        return "".join(["Car with ", str(self.wheels), " wheels and ", str(self.power)])
        
        
        
car = Car(4, 10*W)
car
```


```python
print(car.wheels)
car.wheels = car.wheels*2
print(car.wheels)
car.add_wheels_observer(lambda obj, new:print("new is", new))
print(car.wheels)
car.wheels = car.wheels*2
car.add_wheels_observer(lambda obj, new:print("new power per wheel", obj.power_per_wheel))
car.wheels = car.wheels*2
```


```python

```

```python
%matplotlib qt
```

```python
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 2 * np.pi, 1024)
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

fig, ax = plt.subplots()
im = ax.imshow(data2d)
ax.set_title('Pan on the colorbar to shift the color mapping\n'
             'Zoom on the colorbar to scale the color mapping')
cbar = fig.colorbar(im, ax=ax, label='Interactive colorbar')

def update_cmap(event):
    print(ax.get_xlim())
    print(ax.get_ylim())
    d = im.get_array()
    cbar.update_normal(d)
    #cbar.set_ set_clim(vmin=np.min(d), vmax=np.max(d))    

cid = fig.canvas.mpl_connect('button_release_event', update_cmap)

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)


#plt.show()
```

```python

```

```python
from physipy import Quantity, Dimension, quantify, DimensionError, units
ohm = units["ohm"]

class BoundedQuantity(Quantity):
    
    def __init__(self, *args, min=None, max=None, cb_for_init=None, **kwargs):

        
        self._callbacks = []
        if cb_for_init is not None:
            self._callbacks.append(cb_for_init)
        
        dim = args[1]
        
        if min is None:
            min_v = 0
            min = Quantity(min_v, dim)
        else:
            min = quantify(min)

        if max is None:
            max_v = 100
            max = Quantity(max_v, dim)
        else:
            max = quantify(max)
            
        if not min.dimension == max.dimension:
            raise DimensionError(min.dimension, max.dimension)
        
        if not dim == max.dimension:
            raise DimensionError(min.dimension, max.dimension)
            
        self.min_value = min.value
        self.max_value = max.value
        #self.value = args[0]
        super().__init__(*args, **kwargs)
        
    
    @Quantity.value.setter
    def value(self, value):
        # handle the initial setting value, we can't "get" it yet
        try:
            old = self.value
        except:
            old = "NotDefinedYet"
        if isinstance(value, (list, tuple)):
            self._value = np.array(value)
        else:
            if self.min_value < value < self.max_value:
                self._value = value
            else:
                raise ValueError("Value is not between min and max")
        change_dic = {"old":old, "new":value}
        self.notify_change_in_value(change_dic)    
        
    
    def register_callback_when_value_changes(self, cb):
        """
        Register a callback that will be called on value change.
        """
        self._callbacks.append(cb)
    
    
    def notify_change_in_value(self, change_dic):
        """
        Activate all callbacks that were registered.
        """
        for callback in self._callbacks:
            callback(change_dic)
            
    def as_ipyw(self):
        """
        As slider
        """
        from physipy.qwidgets.qipywidgets import QuantityTextSlider
        s = QuantityTextSlider(self, min=self.min, max=self.max, favunit=self.favunit)
        return s            
                
def bounded_quantity(init, min, max, favunit=None):
    if favunit is None:
        favunit = init.favunit
    return BoundedQuantity(init.value, init.dimension, min=min, max=max, favunit=favunit)
                
R = bounded_quantity((2*ohm).set_favunit(ohm), 0*ohm, 3*ohm)#, favunit=ohm)
print(R)
R.value = 0.2
print(R)
R.register_callback_when_value_changes(lambda _: print("value changed"))
R.value = 0.5
```

```python
from physipy.qwidgets.qipywidgets import QuantityTextSlider
from physipy import set_favunit, units
from physipy.quantity.utils import cached_property_depends_on
V = units["V"]
Farad = units["F"]
```

```python
class Model():
    deps = {}
    
    def __repr__(self):
        return "Model with "+"".join([k+":"+str(v) for k, v in self.pdict.items()])
    
    @classmethod
    def register_dep(cls, base, high):
        cls.deps[high] = [b for b in base]
        
class BestRC(Model):

    def __init__(self, R, C):
        super().__init__()
        self.R = R
        self.C = C
        
    BestRC.register_dep(["R", "C"], "tau")    
    @property
    def tau(self):
        return self.R * self.C

        
    #@depends_on("tau")
    @set_favunit(V)
    def response(self, t):
        return 1 - np.exp(-t/self.tau)
        
       
    @cached_property_depends_on('R', 'C')
    def slow_tau(self):
        time.sleep(5)
        return self.tau**2

        
rc = BestRC(2*ohm, 3*Farad)
rc = BestRC(R={"init":2*ohm, "min":3*ohm},
        C=3*Farad)

print(BestRC.deps)
```

# List of wanted features

<!-- #region -->
- Handle min/max/step/initial value like a slider `
- Handle favunit
- Possibility to add callbacks after init
- Cachable on param state
- Get state of all params as a dict
- Customize __repr__ with all parameters
- Get a Model state that can be saved, with dependency
- Access a slider using _`w` on parameter value


```python
# model inerhit for repr
class BestRC(Model):

    def __init__(self, R, C):
        self.R = 
        self.C = 
    
    #@depends_on("R", "C")
    @property
    def tau(self):
        return self.R * self.C
    register_dep('R', 'C')("tau")
        
    @depends_on("tau")
    @set_favunit(V)
    def response(self, t):
        return 1 - np.exp(-t/self.tau)
        
       
    @cached_property_depends_on('R', 'C')
    def slow_tau(self):
        time.sleep(5)
        return self.tau**2

        
rc = RC(2*ohm, 3*Farad)
rc = RC(C={init=2*ohm, min=3*ohm},
        F=3*Farad)


```
<!-- #endregion -->

```python
def decorator_creator(*param_args):
    
    # define decorator
    def decorator(func):
        def _wrapped_method(self, *method_args, **method_kwargs):
            
            return 
        return _wrapped_method
    return decorator
```

```python

```
