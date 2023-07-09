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
