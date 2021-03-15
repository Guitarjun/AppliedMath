#!/usr/bin/env python
# coding: utf-8

# In[122]:


"""
arj1
Arjun Srivastava
AMATH 301 B
"""

import numpy as np
import matplotlib.pyplot as plt


# In[143]:


# Problem 1

A = np.array([[1, -2, 0, -1], [4, 0, -2, 1]])

# a)

# Reduced SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
V = Vt.copy().T

A1 = V.copy()

# b)

# Full SVD
U, s, Vt = np.linalg.svd(A)
V = Vt.copy().T

A2 = V.copy()

# c)

# Rank
A3 = len(s)

# d)

sigma1 = s[0]
u1 = U[:, 0:1]
v1 = V[:, 0:1]

A4 = sigma1 * u1 @ v1.T

# e)

A5 = (np.cumsum(s ** 2) / np.sum(s ** 2))[0]


# In[144]:


# Problem 2

# Function to generate discrete Poisson matrix of given dimension
def discrete_poisson(dim: int):
    A = np.zeros((dim, dim))
    np.fill_diagonal(A, 2), np.fill_diagonal(A[1:], -1), np.fill_diagonal(A[:, 1:], -1)
    return A

A = discrete_poisson(114)

rho = np.fromfunction(lambda j, i: 2 * (1 - np.cos(53*np.pi / 115)) * np.sin((53*np.pi*(j+1))/115), (114, 1))

# a)

U, s, Vt = np.linalg.svd(A)
V = Vt.T

A6 = s.reshape(1, 114)

# b)

A7 = U.copy().T

# c)

A8 = Vt.copy().T

# d)

S = np.zeros(A.shape)
for i in range(s.size):
    S[i, i] = 1/s[i]
    
A9 = S

# e)

A10 = A7 @ rho

# f)

A11 = A9 @ A10

# g)

A12 = A8 @ A11


# In[145]:


# Problem 3

data = np.genfromtxt('hw10_img.csv', delimiter=',')
mb = 8 / 1e6
h, w = data.shape


# In[153]:


# a)

A13 = h * w * mb

# b)
U, s, Vt = np.linalg.svd(data)

E = np.cumsum(s ** 2) / np.sum(s ** 2)
for k in range(len(s)):
    A14 = k + 1
    if E[k] > 0.99:
        break

# c)

A15 = mb * k * (h + w + 1)


# In[ ]:




