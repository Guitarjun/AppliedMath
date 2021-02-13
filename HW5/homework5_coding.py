#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Arjun Srivastava
arj1
AMATH 301 B
"""

import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt


# In[11]:


# Problem 1

tolerance = 1e-8
x0 = 2

def newton_minima(f, fprime, fdprime, i):
    X = np.zeros(i+1)
    X[0] = x0
    for k in range(i):
        X[k + 1] = X[k] - fprime(X[k]) / fdprime(X[k])
        if np.abs(fprime(X[k + 1])) < tolerance:
            break
    X = X[:(k+2)]
    return np.size(X), X[np.size(X)-1]

# a)

f = lambda x : x**2
fprime = lambda x : 2*x
fdprime = lambda x : 2

A1, A2 = newton_minima(f, fprime, fdprime, 100)

# b)

f = lambda x : x**500
fprime = lambda x : 500 * x**499
fdprime = lambda x : 500 * 499 * x**498

A3, A4 = newton_minima(f, fprime, fdprime, 400)

# c)

f = lambda x : x**1000
fprime = lambda x : 1000 * x**999
fdprime = lambda x : 1000 * 999 * x**998

A5, A6 = newton_minima(f, fprime, fdprime, 750)

# d)

a = -2
b = 2
c = (-1 + np.sqrt(5)) / 2

x = c * a + (1 - c) * b
fx = f(x)
y = (1 - c) * a + c * b
fy = f(y)
for k in range(100):
    if fx < fy:
        b = y
        y = x
        fy = fx
        x = c * a + (1 - c) * b
        fx = f(x)
    else:
        a = x
        x = y
        fx = fy
        y = (1 - c) * a + c * b
        fy = f(y)
    if (b - a) < tolerance:
        k += 2  # Add two initial guesses
        break 

A7 = k + 1
A8 = x


# In[15]:


# Problem 2

c = lambda t : 1.3 * (np.exp(-t/11) - np.exp(-4*t/3))
t = np.arange(0, 30, .1)
plt.title('Concentration of Drug over Time')
plt.xlabel('hours (t)')
plt.ylabel('c(t)')
plt.plot(t, c(t))
plt.grid()

tmax = scipy.optimize.minimize_scalar(lambda t : -c(t), bounds=(0, 10), method='Bounded')
A9 = tmax.x
A10 = c(tmax.x)


# In[22]:


# Problem 3

h = lambda x : -(1/((x-0.3)**2 + 0.01)) - (1/((x-0.9)**2 + 0.04)) - 6
x = np.arange(0, 2, .01)
plt.plot(x, h(x))
plt.title('h(x) vs. x')
plt.xlabel('x')
plt.ylabel('h(x)')
plt.grid()

xmin1 = scipy.optimize.minimize_scalar(h, bounds=(0,0.5), method='Bounded')
A11 = xmin1.x

xmax = scipy.optimize.minimize_scalar(lambda x : -h(x), bounds=(0.5,0.8), method='Bounded')
A12 = xmax.x

xmin2 = scipy.optimize.minimize_scalar(h, bounds=(0.8,2), method='Bounded')
A13 = xmin2.x

print(A12, A13, sep='\n')


# In[111]:


# Problem 4

f = lambda v : (v[0]**2 + v[1] - 11)**2 + (v[0] + v[1]**2 - 7)**2

# a)

xmin = scipy.optimize.minimize(f, np.array([-2, 3]), method='Nelder-Mead')
A14 = xmin.x.reshape((2, 1))

# b)

xmax = scipy.optimize.minimize(lambda v : -f(v), np.array([0, 0]), method='Nelder-Mead')
A15 = xmax.x.reshape((2, 1))


# In[ ]:




