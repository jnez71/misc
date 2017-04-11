"""
Value iteration for a finite markov decision process.

"""
# Dependencies
from __future__ import division
import numpy as np; npl = np.linalg

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
              [ 0,  0, -2]])

# Discount factor
g = 1

# Initial condition and runtime limit
Vlast = np.random.sample(len(S))
imax = 1000000
tol = 1e-5

# Value iteration, heavily vectorized
for i in xrange(imax):
    Q = c + g*np.vstack((P0.dot(Vlast), P1.dot(Vlast)))
    U = np.argmin(Q, axis=0)
    V = Q[U, S]
    if np.allclose(V, Vlast, rtol=tol):
        break
    Vlast = np.copy(V)

# Compute average cost and normalize value function
eta = (V - Vlast)[0]
V = V - V[0] - 1

print("Finished in {} of {} iterations.".format(i+1, imax))
print("Optimal Policy: {}".format(U))
print("Optimal Expected Value Function: {}".format(np.round(V, 3)))
print("Optimal Average Cost: {}".format(np.round(eta, 3)))
