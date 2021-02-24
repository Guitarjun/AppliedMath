#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
arj1
Arjun Srivastava
AMATH 301 B
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


# In[ ]:


# Problem 1

# Population by time
data = np.genfromtxt('population.csv', delimiter=',')
t = data[0, :]
N = data[1, :]

# a)

dx = 10
n = t.size
deriv = np.zeros(n)
deriv[0] = (-3 * N[0] + 4 * N[1] - N[2]) / (2 * dx)
for k in range(1, n - 1):
    deriv[k] = (N[k + 1] - N[k - 1]) / (2 * dx)
deriv[-1] = (3 * N[-1] - 4 * N[-2] + N[-3]) / (2 * dx)

A1, A2, A3, A4 = deriv[-1], deriv[9], deriv[0], deriv.reshape(1, 24)

A5 = (A4 / N).reshape(1, 24)

A6 = np.mean(A5)


# In[2]:


# Problem 2

data = np.genfromtxt('brake_pad.csv', delimiter=',')
r = data[0, :]
T = data[1, :]
n = r.size

dx = r[1] - r[0]

re = 0.308
ro = 0.478
thetaP = 0.7051

# a)

Ttotal = np.zeros(n)
A = np.zeros(n)

for k in range(n):
    Ttotal[k] = T[k] * r[k] * thetaP
    A[k] = r[k] * thetaP
    
A_LHR = dx * np.sum(A[:-1])
A7 = dx * np.sum(Ttotal[:-1])

A8 = A7 / A_LHR


# b)

A_RHR = dx * np.sum(A[1:])
A9 = dx * np.sum(Ttotal[1:])

A10 = A9 / A_RHR

# c)

A11 = (A7 + A9) / 2
A_ = (A_LHR + A_RHR) / 2

A12 = A11 / A_


# In[3]:


# Problem 3

bounds = (0, 1)
F = lambda x : (x**2 / 2) - (x**3 / 3)
mu = (0.95, 0.5, 0.01)

# a)

T = lambda z : mu[0] / np.sqrt(F(mu[0]) - F(mu[0] * z))
A13, err = scipy.integrate.quad(T, bounds[0], bounds[1])

# b)

T = lambda z : mu[1] / np.sqrt(F(mu[1]) - F(mu[1] * z))
A14, err = scipy.integrate.quad(T, bounds[0], bounds[1])

# c)

T = lambda z : mu[2] / np.sqrt(F(mu[2]) - F(mu[2] * z))
A15, err = scipy.integrate.quad(T, bounds[0], bounds[1])

