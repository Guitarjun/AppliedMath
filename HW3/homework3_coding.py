#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
Arjun Srivastava
arj1
AMATH 301 B
"""

import numpy as np
import scipy.linalg


# In[39]:


# Problem 1

A = np.genfromtxt('bridge_matrix.csv', delimiter=',')
b = np.zeros(shape=(13,1))
b[8], b[10], b[12] = 2, 8, 4

# a)

A1 = scipy.linalg.solve(A, b)

# b)

P, L, U = scipy.linalg.lu(A)
P = np.transpose(P)
A2 = L.copy()

# c)

y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
f = scipy.linalg.solve_triangular(U, y)
A3 = y.copy()

# d)

while np.max(np.abs(f)) < 30:
    b[8] += .01
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    f = scipy.linalg.solve_triangular(U, y)
A4 = b[8]
A5 = np.argmax(np.abs(f)) + 1


# In[42]:


# Problem 2

alpha, omega, x0 = -0.003, 0.05, np.transpose(np.array([[1, -1]]))
A = np.array([[1 - alpha, -omega], [omega, 1-alpha]])

# a)

A6, A7 = np.zeros((1, 1001)), np.zeros((1, 1001))
x0 = np.array([[1], [-1]])
for i in range(1001):
    A6[0][i], A7[0][i] = x0[[0]], x0[[1]]
    x0 = scipy.linalg.solve(A, x0)

# b)

A8 = np.zeros((1, 1001))
# x0 = np.array([[1], [-1]])
for i in range(1001):
    A8[0][i] = np.sqrt((A6[0][i]**2 + A7[0][i]**2))
    
# c)

i = 0
while A8[0][i] > 0.05:
    i += 1
A9, A10 = i, A8[0][i]


# In[9]:


# Problem 3

# a)

R = lambda x : np.array([[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]])

A11 = R(np.pi/10)

# b)

x = np.array([[0.3], [1.4], [-2.7]])

A12 = R((3*np.pi)/8) @ x

# c)

y = np.array([[1.2], [-0.3], [2]])
A13 = scipy.linalg.solve(R(np.pi/7), y)

# d)

A14 = scipy.linalg.inv(R((5*np.pi)/7))

# e)

A15 = -((5*np.pi)/7)

