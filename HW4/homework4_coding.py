#!/usr/bin/env python
# coding: utf-8

# In[97]:


"""
Arjun Srivastava
arj1
AMATH 301 B
"""

import numpy as np
import scipy.linalg


# In[161]:


# Problem 1

# a)

# Function to generate discrete Poisson matrix of given dimension
def discrete_poisson(dim: int):
    A = np.zeros((dim, dim))
    np.fill_diagonal(A, 2), np.fill_diagonal(A[1:], -1), np.fill_diagonal(A[:, 1:], -1)
    return A

A = discrete_poisson(114)
A1 = A.copy()

# b)

rho = np.fromfunction(lambda j, i: 2 * (1 - np.cos(53*np.pi / 115)) * np.sin((53*np.pi*(j+1))/115), (114, 1))

A2 = rho.copy()


# In[202]:


# Problem 2

# c)

P = np.diag(np.diag(A1))
T = A1- P
M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)
A3 = np.max(np.abs(w))

# d)

# For

# tolerance = 1e-5
# x0 = np.ones((114, 1))
# X = np.zeros((114, 20001))
# X[:, 0:1] = x0
# solution = np.fromfunction(lambda j, i: np.sin((53*np.pi*(j+1))/115), (114, 1))

# for k in range(20000):
#     X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + rho)
#     if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
#         break
# X = X[:, :(k+2)]
# A4 = X[:, k:k+1]
# A5 = X.shape[1]

# # e)

# A6 = np.max(np.abs(A4-solution))

# While

x0 = np.ones((114, 1))
tolerance = 1e-5
err = tolerance + 1
X = np.zeros((114, 1))
X[:, 0:1] = x0

k = 0
if A3 < 1:
    while err >= tolerance:
        X = np.hstack((X, scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + rho)))
        err = np.max(np.abs(X[:, k+1] - X[:, k]))
        k = k + 1
A4 = X[:, k:k+1]
A5 = X.shape[1]

# e)

solution = np.fromfunction(lambda j, i: np.sin((53*np.pi*(j+1))/115), (114, 1))
A6 = np.max(np.abs(A4-solution))


# In[190]:


# Problem 3

# f)

P = np.tril(A1)
T = A1 - P
M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)
A7 = np.max(np.abs(w))

# g)

# x0 = np.ones((114, 1))
# X = np.zeros((114, 7001))
# X[:, 0:1] = x0

# for k in range(7000):
#     X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + rho, lower=True)
#     if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
#         break
# X = X[:, :(k+2)]
# A8 = X[:, k:k+1]
# A9 = X.shape[1]
# A10 = np.max(np.abs(A8-solution))

err = tolerance + 1
X = np.zeros((114, 1))
X[:, 0:1] = x0

k = 0
if A7 < 1:
    while err >= tolerance:
        X = np.hstack((X, scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + rho, lower=True)))
        err = np.max(np.abs(X[:, k+1] - X[:, k]))
        k = k + 1
A8 = X[:, k:k+1]
A9 = X.shape[1]

# h)

A10 = np.max(np.abs(A8-solution))


# In[191]:


# Problem 4

omega = 1.5
D = np.diag(np.diag(A1))
U = np.triu(A1, 1)
L = np.tril(A1, -1)

# i)

P = ((1/omega) * D) + L
T = (((omega - 1)/omega) * D) + U
A11 = P.copy()

# j)

M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)
A12 = np.max(np.abs(w))

# x0 = np.ones((114, 1))
# X = np.zeros((114, 3001))
# X[:, 0:1] = x0

# for k in range(3000):
#     X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + rho, lower=True)
#     if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
#         break
# X = X[:, :(k+2)]
# A13 = X[:, k:k+1]
# A14 = X.shape[1]
# A15 = np.max(np.abs(A13-solution))

# k)

err = tolerance + 1
X = np.zeros((114, 1))
X[:, 0:1] = x0

k = 0
if A12 < 1:
    while err >= tolerance:
        X = np.hstack((X, scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + rho, lower=True)))
        err = np.max(np.abs(X[:, k+1] - X[:, k]))
        k = k + 1
A13 = X[:, k:k+1]
A14 = X.shape[1]

# l)

A15 = np.max(np.abs(A13-solution))

