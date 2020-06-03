#!/usr/bin/env python2
"""
If you can only measure velocity, how well can you estimate position?
Examine the case of a 1-DOF linear system, using a Kalman filter as an observer.
If acceleration doesn't depend on position, then information cannot flow from
our velocity estimate to our position estimate through the process model. I.e.,
position is unobservable! If our process model is noiseless, as the velocity estimate
converges, the position estimate error will become constant (nonzero). If our process
model is noisy, the position estimate error will fluctuate forever and our position
covariance will increase without bound. However, the cross-covariance remains stable
in any case. This demo was made to clarify something to a friend.

"""
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt

dt = 0.01  # timestep
T = np.arange(0, 100, dt)  # time domain

# State is [position; velocity]
A = np.array([[0,    1],
              [0, -1]])  # pdot = v and vdot = -k*v + u, i.e. damped motion

H = np.array([[0, 1]])  # sensor measures only velocity

U = 2*np.sin(0.5*T)  # excitation acceleration input timeseries, B = I

P = [np.array([[0, 0],
               [0, 2]])]  # initial state covariance and timeseries as a list

Q = np.array([[0,  0],
              [0, 10]])  # process noise, where xdot_true = A*x_true + u + N(0, Q)

R = 2  # sensor noise, where z = H*x_true + N(0, R)
Rinv = 1/R  # memoize...
Rsqrt = np.sqrt(R)

X = [np.array([0, 0])]  # initial state, timeseries list
Xest = [np.array([0, 2])]  # initial state estimate, timeseries list
Z = []; K = []  # for recording measurement and Kalman gain timeseries

# Simulate
for i, t in enumerate(T[1:]):

    Z.append(H.dot(X[i]) + Rsqrt*np.random.randn())  # current noisy measurement
    K.append(P[i].dot(H.T).dot(Rinv))  # current kalman gain

    xestdot = A.dot(Xest[i]) + U[i] + K[i].dot(Z[i] - H.dot(Xest[i]))  # predict + correct
    Xest.append(Xest[i] + xestdot*dt)  # next estimate
    
    Pdot = A.dot(P[i]) + P[i].dot(A.T) + Q - K[i].dot(R).dot(K[i].T)  # Kalman-Bucy covariance derivative (Riccati equation)
    P.append(P[i] + Pdot*dt)  # next state covariance

    xdot = A.dot(X[i]) + U[i] + np.random.multivariate_normal((0, 0), Q)  # true state derivative
    X.append(X[i] + xdot*dt)  # next true state

# Cast lists to arrays
X = np.array(X)
Xest = np.array(Xest)
P = np.array(P)
Z = np.array(Z).flatten()
K = np.array(K)

# Plots
fig = plt.figure()
fig.suptitle("Results", fontsize=20)

ax = fig.add_subplot(3, 1, 1)
ax.set_ylabel("States,\nMeasurements,\n& Estimates", fontsize=14)
ax.plot(T, X[:, 0], c='k', ls='--', label="p true")
ax.plot(T, X[:, 1], c='b', ls='--', label="v true")
ax.plot(T, Xest[:, 0], c='k', label="p est")
ax.plot(T, Xest[:, 1], c='b', label="v est")
ax.scatter(T[1:], Z, c='b', alpha=0.1, s=2, label="meas")
ax.set_xlim([T[0], T[-1]])
ax.grid(True)
ax.legend(loc=1)

ax = fig.add_subplot(3, 1, 2)
ax.set_ylabel("Covariance", fontsize=16)
ax.plot(T, P[:, 0, 0], c='k', label="pp")
ax.plot(T, P[:, 0, 1], c='r', label="pv")
ax.plot(T, P[:, 1, 1], c='b', label="vv")
ax.set_xlim([T[0], T[-1]])
ax.grid(True)
ax.legend(loc=1)

ax = fig.add_subplot(3, 1, 3)
ax.set_ylabel("Kalman Gain", fontsize=16)
ax.plot(T[1:], K[:, 0, 0], c='k', label="p")
ax.plot(T[1:], K[:, 1, 0], c='b', label="v")
ax.set_xlim([T[0], T[-1]])
ax.grid(True)
ax.legend(loc=1)
ax.set_xlabel("Time", fontsize=16)

plt.show()
