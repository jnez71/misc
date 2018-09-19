#!/usr/bin/env python2
"""
Recursive Bayes for POMDP belief-state tracking.
Max-a-posteriori estimation.

"""
# Dependencies
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt

# State, action, measurement, and time cardinalities
nS = 3; nA = 2; nM = 2; nT = 100

# Transition conditional probability matrix, A by S by S'
P = np.array([[[  1,   0,   0],
               [  1,   0,   0],
               [  0, 0.3, 0.7]],
              [[0.4,   0, 0.6],
               [0.1, 0.6, 0.3],
               [  0, 0.1, 0.9]]], dtype=np.float64)

# Sensor conditional probability matrix, A by S by O
Qr = 0.5
Q = np.array([[[   1,  0],
               [   1,  0],
               [1-Qr, Qr]],
              [[   1,  0],
               [   1,  0],
               [   0,  1]]], dtype=np.float64)

# Cost function, c(a,x)
c = np.array([[-1, -1, -3],
              [ 0,  0, -2]], dtype=np.float64)

# State, estimate, measurement, belief, and cost histories
x = np.zeros(nT, dtype=np.int64)
xh = np.zeros(nT, dtype=np.int64)
y = np.zeros(nT, dtype=np.int64)
b = np.zeros((nT, nS), dtype=np.float64)
cost = np.zeros(nT, dtype=np.float64)

# Initial conditions
x[0] = 0
b[0] = [1, 0, 0]

# Function for randomly sampling with a given discrete probability density
sample_from = lambda p: np.argwhere(np.random.sample() < np.cumsum(p))[0][0]

# Simulation
time = np.arange(nT)
for t in time[1:]:

    # Estimate state as the posterior maximizer
    xh[t-1] = np.argmax(b[t-1])

    # Randomly choose action, accept cost
    u = sample_from([0.5, 0.5])
    cost[t] = cost[t-1] + c[u, x[t-1]]

    # Advance state, obtain measurement
    x[t] = sample_from(P[u, x[t-1]])
    y[t] = sample_from(Q[u, x[t]])

    # Update belief
    b[t] = (b[t-1].dot(P[u]))*Q[u, :, y[t]]
    b[t] = b[t] / np.sum(b[t])

# Plot estimation error
print("Accuracy: {}%".format(100*len(np.argwhere(np.logical_not(np.abs(x - xh))))/nT))
plt.title("Estimation Error", fontsize=22)
plt.ylabel("x - xh", fontsize=22)
plt.xlabel("Time (iteration)", fontsize=22)
plt.scatter(time, x-xh)
plt.xlim([0, nT])
plt.grid(True)
plt.show()
