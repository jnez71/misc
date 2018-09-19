#!/usr/bin/env python2
"""
Policy iteration for a finite markov decision process.

"""
# Dependencies
from __future__ import division
import numpy as np; npl = np.linalg
import scipy.linalg as spl

# State and action spaces
S = [0, 1, 2]
A = [0, 1]

# Transition matrix for u=0
P0 = np.array([[  1,   0,   0],
               [  1,   0,   0],
               [  0, 0.3, 0.7]])

# Transition matrix for u=1
P1 = np.array([[0.4,   0, 0.6],
               [0.1, 0.6, 0.3],
               [  0, 0.1, 0.9]])

# Cost matrix, with rows for A and columns for S
c = np.array([[-1, -1, -3],
              [ 0,  0, -2]], dtype=np.float64)

# Initial policy guess and runtime limit
U = np.uint8((np.random.sample(len(S)) > 0.5))
U_last = None
imax = 30

# Discount factor and regularization
g = 1
H = np.zeros_like(P0)
if g == 1: H[:, 0] = 1

# Policy iteration
converged = False
for i in xrange(imax):
    print("Policy iteration: {}".format(i+1))
    Pu = np.array([p1 if u else p0 for p0, p1, u in zip(P0, P1, U)])
    cu = c[U, S]
    print("Solving poisson...")
    V = npl.solve(g*Pu - np.eye(len(Pu)) - H, -cu)
    Q = c + g*np.vstack((P0.dot(V), P1.dot(V)))
    U = np.argmin(Q, axis=0)
    print("Expected average cost: {}\n".format(np.round(V[0], 5)))
    if U_last is not None and np.all(U == U_last):
        converged = True
        break
    U_last = np.copy(U)

# Compute average cost and normalize value function
eta = npl.matrix_power(Pu, 1000)[0, :].dot(cu)
V = V - V[0] - 1

if converged:
    print("Converged!")
    print("Optimal Policy: {}".format(U))
    print("Optimal Expected Value Function: {}".format(np.round(V, 3)))
    print("Optimal Average Cost: {}\n".format(np.round(eta, 3)))
