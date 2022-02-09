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

https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a


The heat equation is basicaly a partial differential equation that mixes time and space : 


$$ \frac{\partial u}{\partial t} - \alpha \nabla u = 0 $$



with $\alpha$ a diffusivity constant. More precisely in 2D : 


$$\frac{\partial u}{\partial t} - \alpha \left( \frac{\partial^2 u}{\partial x^2} +  \frac{\partial^2 u}{\partial y^2}  \right) = 0$$



Using Finite-Difference method simply consist in approximating the derivatives using small differences between values at small samples.


Using finite-difference, we can rewrite the 2D heat equation : 


$$\frac{u_{i,j}^{k+1} - u_{i,j}^k}{\Delta t} - \alpha \left( \frac{u_{i+1,j}^k - 2 u_{i,j}^k + u_{i-1,k}^k}{\Delta x^2} + \frac{u_{i,j+1}^k - 2 u_{i,j}^k + u_{i,k-1}^k}{\Delta y^2}  \right) = 0$$


Suppose $\Delta x = \Delta y$, we can get : 
$$u_{i,j}^{k+1} = \alpha\frac{\Delta t }{\Delta x^2}\left(u_{i+1,j}^k + u_{i-1,j}^k + u_{i,j+1}^k + u_{i,j-1}^k - 4 u_{i,j}^k  \right)+ u_{i,j}^k$$


For numerical stability, we need : 
$$\Delta t \le \frac{\Delta x^2}{4\alpha}$$


Now about optimization of the loops : 
notice that the equation to compute the temperature at time k+1 is a linear combination for other temperature points at time k. So this relation can be seen as a linear operation, and so can be writter with a convolution kernel. Picture the heat map at time k as a 2D image, and the heat map at time k+1 as another image that is the result of a convolution of the first image.
Rewriting the equation with $\gamma = \alpha \frac{\Delta t}{\Delta x^2}$, we get : 
$$u_{i,j}^{k+1} = \gamma u_{i+1,j}^k + \gamma u_{i-1,j}^k + \gamma u_{i,j+1}^k + \gamma u_{i,j-1}^k - 4 \gamma u_{i,j}^k +u_{i,j}^k$$
The kernel can be seen as  :
$$K = \begin{pmatrix}
0 & \gamma & 0\\
\gamma & 1-4\gamma & \gamma\\
0 & \gamma & 0\\
\end{pmatrix}
$$
with local heatmap =
$$\begin{pmatrix}
u_{i-1,j-1}^k & u_{i-1,j}^k & u_{i-1,j+1}^k\\
u_{i,j-1}^k & u_{i,j}^k & u_{i,j+1}^k\\
u_{i+1,j-1}^k & u_{i+1,j}^k & u_{i+1,j+1}^k\\
\end{pmatrix}
$$



We can then rewrite the equation of $u_{i,j}^{k+1}$ as a simple dot product


Now since the kernel is the same for all element

```python
submaps = np.lib.stride_tricks.sliding_window_view(arr, (3,3))
kernel = np.array([
    [0, gamma, 0],
    [gamma, 1-4*gamma, gamma],
    [0, gamma, 0],
])

print(kernel * submaps)
print(np.sum(kernel * submaps, axis=(2,3)))

```

```python
nx = 4
ny = 4

arr = np.arange(nx*ny).reshape((nx,ny))
print(arr)
itemsize = arr.itemsize
shape = (
    # shape of the output dimension is 
    (nx-2)*(ny-2),
    3,
    3
)
strides = (
    # adjacent elements in the output were originaly XX bytes apart in the input
    8, 8*nx, 8,
)
np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
```

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

print("2D heat equation solver")

plate_length = 200
max_iter_time = 750

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)
kernel = np.array([
    [0, gamma, 0],
    [gamma, 1-4*gamma, gamma],
    [0, gamma, 0],
])


def convolv(sub_arr, kernel=kernel):
    submaps = np.lib.stride_tricks.sliding_window_view(sub_arr, (3,3))
    result = np.sum(kernel * submaps, axis=(2,3))
    return result

def initialize_u():
    # Initialize solution: the grid of u(k, i, j)
    u = np.empty((max_iter_time, plate_length, plate_length))

    # Initial condition everywhere inside the grid
    u_initial = 0

    # Boundary conditions
    u_top = 100.0
    u_left = 0.0
    u_bottom = 0.0
    u_right = 0.0

    # Set the initial condition
    u.fill(u_initial)

    # Set the boundary conditions
    u[:, (plate_length-1):, :] = u_top
    u[:, :, :1] = u_left
    u[:, :1, 1:] = u_bottom
    u[:, :, (plate_length-1):] = u_right

    return u
    
def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]
    return u

def calculate_faster(u):
    for k in range(0, max_iter_time-1, 1):
        # we get a 4D array that contains all possible 3x3 local heatmaps at time k
        local_maps = np.lib.stride_tricks.sliding_window_view(u[k], (3,3))
        # sum the product of the kernel and each map
        # and sum each local map
        result = np.sum(kernel * submaps, axis=(2,3))
        # set the newly computed heatmap at time k+1
        u[k+1, 1:-1, 1:-1] = res
    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()


    return plt
```

```python

u1 = calculate(u)
```

```python
u2 = calculate_faster(u)
```

```python
np.all(u1 == u2)
```

```python
# Do the calculation here
u = calculate(u)

def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("heat_equation_solution.gif")

print("Done!")
```

# Using implicit method


https://yaredwb.github.io/FDM1D/


We can rewrite the time derivative using a backward difference, which makes the method implicit : 
$$\frac{\partial u}{\partial t} \approx \frac{u_{i,j}^{k-1} - u_{i,j}^k}{\Delta t}$$


where it was : 
$$\frac{u_{i,j}^{k+1} - u_{i,j}^k}{\Delta t}$$


So the full equation becomes :
$$ \frac{u_{i,j}^{k-1} - u_{i,j}^k}{\Delta t} - \alpha \left( \frac{u_{i+1,j}^k - 2 u_{i,j}^k + u_{i-1,k}^k}{\Delta x^2} + \frac{u_{i,j+1}^k - 2 u_{i,j}^k + u_{i,j-1}^k}{\Delta y^2}  \right) = 0$$


and knowing the heatmap at time $k-1$, trying to solve for $u_{i,j}^k$ we get a dependence from neigbouhring samples also at time $k$, which we don't know. This equation is true for all samples $i,j$, so we can write it for all samples, and we get a system of N-samples equations that link all heat points of time $k$.
This system of equation is linear and can be solved using classic linear algebra to inverse the matrix of the system.


We write the full the full equation for 3 samples in 1D:



$$ \frac{u_{i}^{k-1} - u_{i}^k}{\Delta t} - \alpha \left( \frac{u_{i+1}^k - 2 u_{i}^k +u_{i-1}^k}{\Delta x^2}  \right) = 0$$
so


$$u_i^{k-1} - u_i^k = \alpha\frac{\Delta t}{\Delta x^2} \left( u_{i+1}^k - 2 u_{i}^k +u_{i-1}^k \right)$$


$$u_i^k (2 \gamma - 1 ) = \gamma u_{i+1}^k + \gamma u_{i-1}^k - u_{i}^{k-1}$$


Supposing the bound condition at i=0 and i=4 is known, we have for the central 3 samples  : 
$$u_1^k (2 \gamma - 1 ) = \gamma u_{2}^k + \gamma u_{0}^k - u_{1}^{k-1}$$
$$u_2^k (2 \gamma - 1 ) = \gamma u_{3}^k + \gamma u_{1}^k - u_{2}^{k-1}$$
$$u_3^k (2 \gamma - 1 ) = \gamma u_{4}^k + \gamma u_{2}^k - u_{3}^{k-1}$$


remember that only the samples at time k-1 are known.


which can be written is matrix form : 


$$\begin{pmatrix}

\end{pmatrix}$$


# Crank-Nicholson method


This method is a combination of the explicit and implicit methods. The time derivative is approximated using a central difference equation. At time k+1/2 :
$$ \frac{\partial u}{\partial t} \approx \frac{u_i^{k+1} - u_i^k}{\Delta t}$$


For spatial at time k+1/2 : 
$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1}^{k+1/2} - 2u_i^{k+1/2} + u_{i-1}^{k+1/2}}{\Delta z^2} $$
Now we approximate the value at half time samples as the mean of surounding time samples : 

$$u_i^{k+1/2} = \frac{1}{2}\left( u_i^k + u_i^{k+1} \right)$$
So replacing in the above equaiton : 


$$\frac{\partial^2 u}{\partial x^2} \approx \frac{\frac{1}{2}\left( u_{i+1}^k + u_{i+1}^{k+1} \right) - 2\frac{1}{2}\left( u_i^k + u_i^{k+1} \right) + \frac{1}{2}\left( u_{i-1}^k + u_{i-1}^{k+1} \right)}{\Delta z^2} $$



SPlitting time k and k+1 : 


$$\frac{\partial^2 u}{\partial x^2} \approx \frac{1}{2} \left( \frac{u_{i+1}^{k+1} -2 u_i^{k+1} + u_{i-1}^{k+1}}{\Delta z^2} + \frac{u_{i+1}^{k} -2 u_i^{k} + u_{i-1}^{k}}{\Delta z^2}\right)$$


So the global PDE becomes : 


$$ \frac{u_i^{k+1} - u_i^k}{\Delta t} - \frac{\alpha}{2} \left( \frac{u_{i+1}^{k+1} -2 u_i^{k+1} + u_{i-1}^{k+1}}{\Delta z^2} + \frac{u_{i+1}^{k} -2 u_i^{k} + u_{i-1}^{k}}{\Delta z^2}\right) = 0$$


Like the implicit method, the Crank-Nicolson method requires solving a system of equations at each time step since the unknown un+1i is coupled with its neighboring unknowns un+1i−1 and un+1i+1. The same principle can be used to propagate the system as with the implicit method and the linear system solving using matrix.

```python

```
