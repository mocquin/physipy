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

# Using scipy


## Introduction : without units


Use the well known RC circuit as a use case : see [this post](https://medium.com/towards-data-science/interactive-plotting-the-well-know-rc-circuit-in-jupyter-d153c0e9d3a).

```python
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np
 
# in Ohms
R = 10000
# in Farad
capa = 1*10**-12
# time constant
tau= R*capa
# Source in volts
Ve = 1
# initial tension in volts
y0 = [0]
 
def analytical_solution(t):
    return (y0[0]-Ve)*np.exp(-t/tau) + Ve
 
def source_tension(t):
    return Ve
 
def RHS_dydt(t, y):
    return 1/(tau)*(source_tension(t) - y)
 
t_span = (0, 10*tau)

solution = scipy.integrate.solve_ivp(
    RHS_dydt,
    t_span,
    y0,
    dense_output=True,
)

# lets visualize the solver's solution along
# with the analytical solution
fig, ax = plt.subplots()
ech_t = np.linspace(0, 10*tau)

ax.plot(ech_t,
        solution.sol(ech_t)[0],
        "-o", 
        label="Solver's solution")
ax.plot(ech_t,
        analytical_solution(ech_t),
        label="Analytical solution")
ax.legend()
solution
```

## With units
We use the convenience function wrapped `from physipy.integrate import solve_ivp`

```python
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np
 
from physipy import units, s, set_favunit, setup_matplotlib
from physipy.integrate import solve_ivp
    
setup_matplotlib()
    
ohm = units["ohm"]
F = units["F"]
V = units["V"]

# in Ohms
R = 10000 * ohm
# in Farad
capa = 1*10**-12 * F
# time constant
tau= R*capa
# Source in volts
Ve = 1 * V
# initial tension in volts
y0 = [0*V]

@set_favunit(V)
def analytical_solution(t):
    return (y0[0]-Ve)*np.exp(-t/tau) + Ve
 
def RHS_dydt(t, y):
    return 1/(tau)*(Ve - y)
 
t_span = (0*s, 10*tau)

solution = solve_ivp(
    RHS_dydt,
    t_span,
    y0,
    dense_output=True,
)
```

```python
solution.t
```

```python
solution.y[0]
```

```python
solution.sol(1)
```

```python
# lets visualize the solver's solution along
# with the analytical solution
fig, ax = plt.subplots()
ech_t = np.linspace(0*s, 10*tau)

ax.plot(ech_t,
        solution.sol(ech_t)[0],
        "-o", 
        label="Solver's solution")
ax.plot(ech_t,
        analytical_solution(ech_t),
        "--",
        label="Analytical solution")
ax.legend()
```

```python
print((solution.sol(ech_t)[0]/ analytical_solution(ech_t)-1)*100)
```

## Solve multiple equations


We create various systems with different parameters and use numpy broadcasting : 
 - several resistance
 - several capa
 
The only modification must be : 
 - t_span is the same for all systems, so still a 2-uple
 - must specify initial condition for each system, so y0 is now a N-iterable (for N systems)

```python
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np
 
from physipy import units, s, set_favunit, setup_matplotlib
from physipy.integrate import solve_ivp
    
setup_matplotlib()
    
ohm = units["ohm"]
F = units["F"]
V = units["V"]

Rs_, capas_ = np.meshgrid(10000 * ohm + np.linspace(-1000, +1000, num=3)*ohm, 
                          1*10**-12 * F + np.linspace(-0.2, 0.2, num=5)*10**-12*F,
                         )
# in Ohms
R = Rs_.flatten()
R.favunit = ohm
# in Farad
capa = capas_.flatten()
capa.favunit = F
# time constant
tau= R*capa
# Source in volts
Ve = 1 * V
# initial tension in volts
y0 = np.ones_like(R)* 0*V

def RHS_dydt(t, y):
    return 1/(tau)*(Ve - y)
 
t_span = (0*s, 5*np.max(tau))

solution = solve_ivp(
    RHS_dydt,
    t_span,
    y0,
    dense_output=True,
    t_eval=np.linspace(0*s, 5*np.max(tau), num=100),
)
solution
```

```python
@set_favunit(V)
def analytical_solution(t):
    return (y0-Ve)*np.exp(-t/tau) + Ve
 
ns = units["ns"]
tau.favunit = ns
    
analytical_solutions = analytical_solution(solution.t[:, np.newaxis]).T

fig, ax = plt.subplots(figsize=(12, 8))
for i in range(len(R)):
    ax.plot(solution.t,
            solution.y[i],
            label=f"Numercial Tau={tau[i]:.2f}")
    ax.plot(solution.t,
            analytical_solutions[i], 
            "o", 
            label=f"Analytical Tau={tau[i]:.2f}")
ax.legend()
```

## Performance comparison

```python
def timeit_without_unit():
    import scipy.integrate
    import numpy as np
     
    # in Ohms
    R = 10000
    # in Farad
    capa = 1*10**-12
    # time constant
    tau= R*capa
    # Source in volts
    Ve = 1
    # initial tension in volts
    y0 = [0]

    def RHS_dydt(t, y):
        return 1/(tau)*(source_tension(t) - y)
     
    t_span = (0, 10*tau)
    
    %timeit scipy.integrate.solve_ivp(RHS_dydt, t_span, y0, dense_output=True)
    %timeit scipy.integrate.solve_ivp(RHS_dydt, t_span, y0, dense_output=False)
    

def timeit_with_unit():
    import numpy as np
    from physipy.integrate import solve_ivp     
    from physipy import units, s, set_favunit, setup_matplotlib
        
    setup_matplotlib()
        
    ohm = units["ohm"]
    F = units["F"]
    V = units["V"]
    
    # in Ohms
    R = 10000 * ohm
    # in Farad
    capa = 1*10**-12 * F
    # time constant
    tau= R*capa
    # Source in volts
    Ve = 1 * V
    # initial tension in volts
    y0 = [0*V]
     
    def RHS_dydt(t, y):
        return 1/(tau)*(Ve - y)
     
    t_span = (0*s, 10*tau)

    %timeit solve_ivp(RHS_dydt, t_span, y0, dense_output=True)
    %timeit solve_ivp(RHS_dydt, t_span, y0, dense_output=False)

    
timeit_without_unit()
timeit_with_unit()
```

So, yeah, do not use units if performance is important...

```python

```

```python

```

# Numerical methods


## Explicit-forward euler, or rectangle from the left


Called explicit because the value of $y_{t+dt}$ is computed using the current-know value of $y(t)$
$$y_{t+dt} - y_t = \int_{t}^{t+dt}f(t, y(t))df \approx dt f(t, y(t)) $$
hence 
$$y_{t+dt} \approx y_t  + dt f(t, y(t)) $$

This method has error $e_1 = \frac{1}{2}dt (dt\cdot slope) = \frac{dt^2}{2}slope$ since the error is the rectangle-triangle with base dt and side $dt slope$. Cumulating on N segments thats divides the total time T with dt sub-segments $N=T/dt$, with get a cumulated error of $E=Ne_1=T/dt \frac{dt^2}{2}slope = \frac{T}{2}dt slope$, hence proportional to dt : we say that the error is of order 1.


## Implicit-backward euler, or rectangle from the right


Called implicit because the value of $y_{t+dt}$ is computed using the same value  $y_{t+dt}$ : it requires the use of another algorithm to resolve the unknown
$$y_{t+dt} - y_t = \int_{t}^{t+dt}f(t, y(t))df \approx dt f(t+dt, y(t+dt))$$
The error of this method is the opposite of that of the explicit method, hence also of order 1.


## Trapezoidal


Mix between left and right rectangle


$$y_{t+dt} - y_t \approx dt \frac{1}{2}\left( f(t, y(t)) + f(t+dt, y(t+dt)) \right)$$


For now this equation is implicit, because $y_{t+dt}$ appears on both sides. We can turn it into explicit using : 
$$y(t+dt) \approx y(t) + dt f(t,y(t))$$
hence
$$y_{t+dt} - y_t \approx dt \frac{1}{2}\left( f(t, y(t)) + f(t+dt, y(t) + dt f(t,y(t))) \right)$$


This method introduces 2 sources of errors : 


## Runge-Kutta



 - https://medium.com/geekculture/runge-kutta-numerical-integration-of-ordinary-differential-equations-in-python-9c8ab7fb279c
 - https://www.youtube.com/watch?v=2vslKRPlgo0
 - https://www.youtube.com/watch?v=r-jWnXjwQvk
 - https://www.youtube.com/watch?v=5CXhHx56COo&t=1650s
 - https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
 - https://www.physagreg.fr/methodes-numeriques/methodes-numeriques-euler-runge-kutta.pdf
 - https://femto-physique.fr/analyse-numerique/runge-kutta.php


This method is basically equivalent to Simpson's integral method, that approximate the function by a second-order polynom, ie a parabola, that has same values at bound and middle point.


According to scipy doc : 
Explicit Runge-Kutta methods (‘RK23’, ‘RK45’, ‘DOP853’) should be used for non-stiff problems and implicit methods (‘Radau’, ‘BDF’) for stiff problems [9]. Among Runge-Kutta methods, ‘DOP853’ is recommended for solving with high precision (low values of rtol and atol).


The value of the function is estimated using a certain slope coefficient :
$$y_(k+1) = y_k + dt  \phi(t, y, dt)$$
where $\phi(t, y, dt)$ is a mean slope of y between $t_k$ and $t_{k+1}$.
Order-4 Runge-Kutta needs 4 computations with the first derivative 
- f1 = f(t_k, y_k) : derivative at current point, slope at beginning of segment
- f2 = f(t_k+dt/2, y_k+dt/2 f1) 
- f3 = f(t_k+dt/2, y_k+dt/2 f2) 
- f4 = f(t_k+dt, y_k+dt   f3) 
Then compute mean slope : 
$$y_(k+1) = y_k + \frac{dt}{6} (f1 + 2f2 + 2f3 + f4)$$
The error is of order $t^5$ and cumulated error on N segments is $dt^4$

```python

def rk4(func, tk, _yk, _dt=0.01, **kwargs):
    """
    single-step fourth-order numerical integration (RK4) method
    func: system of first order ODEs
    tk: current time step
    _yk: current state vector [y1, y2, y3, ...]
    _dt: discrete time step size
    **kwargs: additional parameters for ODE system
    returns: y evaluated at time k+1
    """

    # evaluate derivative at several stages within time interval
    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    # return an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

```
