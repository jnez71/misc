#!/usr/bin/env python3
"""
Value iteration for a finite markov decision process.
Solving the "badger problem":
    There are n crates and b badgers. The other n-b crates
    have $1 each in them. If you open a badger crate, you lose
    all your earnings. When should you stop opening crates?

"""
# Dependencies
from __future__ import division
import numpy as np; npl = np.linalg

# Crates and badgers
n = 5
b = 1
print("Crates: {}".format(n))
print("Badgers: {}".format(b))

# State and action spaces
S = np.arange(n+2-b)
A = np.arange(2)

# Transition matrix for u=0
P0 = np.eye(len(S), dtype=np.float64)

# Transition matrix for u=1
P1 = np.zeros((len(S), len(S)), dtype=np.float64)
P1[:, -1] = 1.0
for i in S[:-1]:
    l = n - i
    P1[i, i+1] = (l-b)/l
    P1[i, -1] = 1.0 - P1[i, i+1]

# Reward matrix for u=0
R0 = np.zeros((len(S), len(S)), dtype=np.float64)

# Reward matrix for u=1
R1 = np.zeros((len(S), len(S)), dtype=np.float64)
for i in S[:-1]:
    R1[i, i+1] = 1.0
    R1[i, -1] = -i

# Reward expectations
r = np.vstack((np.average(R0, weights=P0, axis=1),
               np.average(R1, weights=P1, axis=1)))

# Discount factor
g = 0.999

# Initial condition and runtime limit
Vlast = np.random.sample(len(S))
imax = 100000
tol = 1e-5

# Value iteration, heavily vectorized
print("Calculating...")
for i in range(imax):
    Q = r + g*np.vstack((P0.dot(Vlast),
                         P1.dot(Vlast)))
    U = np.argmax(Q, axis=0)
    V = Q[U, S]
    if np.allclose(V, Vlast, rtol=tol):
        break
    Vlast = np.copy(V)

# Extract and verify stopping decision
choice = np.count_nonzero(U)
assert np.all(U[:choice])

# Show results
print("(...finished @ iter {} / {})".format(i, imax))
print("Optimal Stop Policy: {}".format(choice))
print("Optimal Expected Value: {}".format(np.round(V[0], 3)))
