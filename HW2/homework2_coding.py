#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Arjun Srivastava
arj1
AMATH 301 B
"""

import numpy as np


# In[199]:


# Problem 1

A = np.fromfunction(lambda i, j: 1 / ((i+1) * (j+1)), (19, 34))  # Using a NumPy function instead of nested for loops

# a)

A1 = A.copy()

# b)

B = A.copy()
B[16,:], B[:,32] = 0, 0
A2 = B.copy()

# c)

A3 = A2[-5:, -3:].copy()

#d)

A4 = B[:, 4].copy().reshape(19,1)


# In[176]:


# Problem 2

# Function to calculate nth term of harmonic series
def harmonic(n):
    res = 0
    for i in range(1, n+1):
        res += 1/i
    return res


# Function to calculate smallest term n where S(n) < i
def harmonic_smallest(i):
    res = 0
    n = 1
    while res <= i:
        res += 1/n
        n += 1
    return n - 1, res

# a)

A5 = harmonic(10)

# b)

A6 = harmonic(100)

# c)

A7, A8 = harmonic_smallest(5)

# d)

A9, A10 = harmonic_smallest(15)


# In[184]:


# Problem 3

# Computes a single log map iteration
def log_map(r, p, K):
    return r * p * (1 - p/K)

# Computes nth log map iteration
def calculate_log(n, r, p, K):
    for i in range(n):
        p = log_map(r, p, K)
    return p

# a)

A11 = np.zeros((1, 3))

for j in range(3):
    log = calculate_log(j+998, 2.5, 3, 20)
    A11[0, j] = log

# b)

A12 = np.zeros((1, 3))

for j in range(3):
    log = calculate_log(j+998, 3.2, 3, 20)
    A12[0, j] = log

# c)

A13 = np.zeros((1, 3))

for j in range(3):
    log = calculate_log(j+998, 3.5, 3, 20)
    A13[0, j] = log

