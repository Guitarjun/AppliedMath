#!/usr/bin/env python
# coding: utf-8

# In[96]:


"""
arj1
Arjun Srivastava
AMATH 301 B
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize


# In[167]:


# Problem 1

x_true = lambda t : .5 * (np.cos(t) + np.sin(t) + np.exp(-t))

# a) Forwards Euler

x0 = 1
dt = 0.1
t = np.arange(0, 10 + dt, dt)

n = t.size
x = np.zeros(n)
x[0] = x0
for k in range(n-1):
    x[k + 1] = x[k] + dt * (np.cos(t[k]) - x[k])

A1 = x.copy().reshape(1, n)

# b) Forwards Euler Error

A2 = np.zeros((1, n))
for k in range(n):
    A2[0, k] = np.abs(x[k] - x_true(t[k]))

# c) Backwards Euler

for k in range(n-1):
    x[k + 1] = (x[k] + dt * (np.cos(t[k+1]))) / (1+dt)
    
A3 = x.copy().reshape(1, n)

# d) Backwards Euler Error

A4 = np.zeros((1, n))
for k in range(n):
    A4[0, k] = np.abs(x[k] - x_true(t[k]))
    
# e) solve_ivp

f = lambda t, x : np.cos(t) - x
tspan = (0, 10)
x0 = np.array([1])
sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
A5 = sol.y[0, :].reshape(1, n)

# f) solve_ivp error

A6 = np.abs(A5 - x_true(t))


# In[186]:


# Problem 2

dxdt = lambda t, x : 8 * np.sin(x)
x_true = lambda t : 2 * np.arctan(np.exp(8*t) / (1 + np.sqrt(2)))
x0 = np.pi / 4

# a) Forwards Euler

dt = 0.01
t = np.arange(0, 2 + dt, dt)

n = t.size
x = np.zeros(n)
x[0] = x0
for k in range(n-1):
    x[k + 1] = x[k] + dt * 8 * np.sin(x[k])
    
A7 = x.copy().reshape(1, n)

# b) Forwards Euler Max Error

A8 = np.max(np.abs(x - x_true(t)))
    
# c) Forwards Euler 2

dt = 0.001
t = np.arange(0, 2 + dt, dt)

n = t.size
x = np.zeros(n)
x[0] = x0

for k in range(n-1):
    x[k + 1] = x[k] + dt * 8 * np.sin(x[k])
    
# Forwards Euler 2 Max Error Ratio
err_001 = np.max(np.abs(x - x_true(t)))
A9 = A8 / err_001

# d) Backwards Euler

dt = 0.01
t = np.arange(0, 2 + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
z0 = 3

for k in range(n-1):
    f = lambda x1 : x1 - x[k] - 8 * dt * np.sin(x1)  # x1 = x[k+1]
    z = scipy.optimize.fsolve(f, z0)
    x[k + 1] = z
    
A10 = x.copy().reshape(1, n)
    
# e) Backwards Euler Max Error

A11 = np.max(np.abs(x - x_true(t)))

# f) error ratio

dt = 0.001
t = np.arange(0, 2 + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0

for k in range(n-1):
    f = lambda x1 : x1 - x[k] - 8 * dt * np.sin(x1)
    z = scipy.optimize.fsolve(f, z0)
    x[k + 1] = z
    
err_001 = np.max(np.abs(x - x_true(t)))
A12 = A11 / err_001

# g) solve_ivp

tspan = (0, 2)
dt = 0.01
t = np.arange(0, 2 + dt, dt)
n = t.size
x0 = np.array([np.pi/4])
sol = scipy.integrate.solve_ivp(dxdt, tspan, x0, t_eval=t)
x = sol.y[0, :]

A13 = x.copy().reshape(1, n)

# h) solve_ivp max error

A14 = np.max(np.abs(A13 - x_true(t)))

# i) solve_ivp 2

tspan = (0, 2)
dt = 0.001
t = np.arange(0, 2 + dt, dt)
n = t.size
x0 = np.array([np.pi/4])
sol = scipy.integrate.solve_ivp(dxdt, tspan, x0, t_eval=t)
x = sol.y[0, :]

err_001 = np.max(np.abs(x - x_true(t)))

A15 = A14 / err_001

