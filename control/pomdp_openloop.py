#!/usr/bin/env python3
"""
Demonstration that:
    1. In stochastic control, optimal open-loop and closed-loop policies can differ
    2. For POMDPs (belief MDPs), the optimal open-loop value-function is linear
    3. ^ and invariant to the measurement model, while closed-loop is not

"""
import numpy as np
import matplotlib.pyplot as plt

npr = np.random
npr.seed(0)

################################################################################

# Cardinalities
nS = 3  # states
nA = 4  # actions
nM = 2  # measurements
nT = 10  # times
nI = 100000  # samples for averages

# Process model
P = npr.uniform(size=(nA, nS, nS))
P /= P.sum(axis=2)[:, :, None]

# Measurement model
Q = npr.uniform(size=(nS, nM))
Q /= Q.sum(axis=1)[:, None]

# Reward function
R = 100*npr.uniform(size=(nA, nS))
R[1:,:] = R[0, :]  # action-independent

# Policy (comment the returns to see different results)
def U(b, t):
    return int(nA*np.cos(100*(b.max()-b.min())))  # closed-loop (depends on belief)
    return np.mod(t, nA)  # open-loop (depends only on time)

################################################################################

# Belief trajectory / reward, for no measurements (deterministic)
E = np.empty((nT, nS), float)
E[0] = npr.uniform(size=nS)  # random initial condition
E[0] /= E[0].sum()
E_R = 0.0
for t in range(1, nT):
    u = U(E[t-1], t-1)
    E[t] = E[t-1].dot(P[u])
    E_R += R[u].dot(E[t])

# Belief trajectory / reward samples, under measurements
B = np.empty((nI, nT, nS), float)
B_R = np.zeros(nI, float)
for i in range(nI):
    if np.mod(100*i/nI, 10) == 0:
        print(f"{int(100*i/nI)}%")
    B[i,0] = E[0]  # same initial condition
    for t in range(1, nT):
        u = U(B[i,t-1], t-1)
        y = npr.choice(nM, p=B[i,t-1].dot(P[u]).dot(Q))
        B[i,t] = B[i,t-1].dot(P[u]) * Q[:,y]
        B[i,t] /= B[i,t].sum()
        B_R[i] += R[u].dot(B[i,t])  # pure POMDP reward, not active-perception

################################################################################

# Difference between no-measurement-value and average-value for increasing sample count
NI = np.arange(1, nI+1)
D_R = E_R - np.cumsum(B_R)/NI

plt.plot(NI, D_R, color='red')
plt.plot(NI, np.zeros(nI), color='blue')

plt.xlabel("nI", fontsize=18)
plt.ylabel("E_R - mean(B_R[:nI])", fontsize=18)

plt.grid(True)
plt.show()

################################################################################
