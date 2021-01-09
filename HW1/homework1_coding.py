#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
Arjun Srivastava
arj1
AMATH 301 B
"""

import numpy as np


# In[5]:


# Problem 1

A = np.array([[1, -2.3], [4.5, np.exp(2)]])
B = np.array([[6, 2, 4, -3], [np.pi, 9, 3.6, -2.1]])
C = np.array([[3.7, -2.4, 0], [4, 1.8, -11.6], [2.3, -3.3, 5.9]])
x = np.array([[5], [np.sin(4)], [-3]])
y = np.array([8, -6])
z = np.array([[3], [0], [np.tan(2)], [-4.7]])

# a)

A1 = 3 * x

# b)

A2 = (z.transpose().dot(B.transpose())) + y

# c) 

A3 = C.dot(x)

# d)

A4 = A.dot(B)

# e)

A5 = B.transpose().dot(A.transpose())


# In[19]:


# Problem 2

# a)

A6 = np.linspace(-4, 1, num=73).reshape(1, 73)

# b)

A7 = np.cos(np.arange(73)).reshape(1, 73)

# c)

A8 = (A6*A7).reshape(1, 73)

# d)

A9 = (A6/A7).reshape(1, 73)

# e)

A10 = (A6**3 - A7).reshape(1, 73)


# In[10]:


# Problem 3

# Computes a single log map iteration
def log_map(r, p, K):
    return r * p * (1 - p/K)

# Computes nth log map iteration
def calculate_log(n, r, p, K):
    for i in range(n):
        p = log_map(r, p, K)
    return p

# Computes a single ricker map iteration
def ricker_map(r, p, K):
    return p * np.exp(r * (1 - p/K))

# Computes nth ricker map iteration
def calculate_ricker(n, r, p, K):
    for i in range(n):
        p = ricker_map(r, p, K)
    return p

# a)

A11 = calculate_log(3, 2.5, 3, 20)

# b)

A12 = calculate_log(4, 3.2, 8, 14)

# c)

A13 = calculate_ricker(3, 2.6, 5, 12)

# d)

A14 = calculate_ricker(4, 3, 2, 25)

# e

A15 = calculate_ricker(500, 3.1, 0, 20)


# In[ ]:




