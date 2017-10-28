#!/usr/bin/python
"""
Formulating the finite-horizon discrete-time linear control problem
as an algebraic minimization problem, which can then be solved with
an arbitrary cost function and hard constraints by a nonlinear solver.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.optimize import minimize
from matplotlib import pyplot

# Discretization and time horizon of interest
dt = 0.05
T = np.arange(0, 10, dt)
n_t = len(T)

# A = df/dx, x_next = f(x, u) ~= A*x + B*u
n_x = 2
A = np.array([[  1,   dt],
              [-dt, 1-dt]])

# B = df/du
n_u = 1
B = np.array([[ 0],
              [dt]])

# Control limits
u_lims = (-10, 10)

# Efficiently compute the matrix powers of A and the simulation matrix L
# x(t) = L*u(t) + Apows*x(0),  L is lower triangular (system is Markovian)
Lb = [np.zeros((n_x, n_u))]
L = np.zeros((n_x*n_t, n_u*n_t))
Apows = [np.eye(n_x)]
for i in xrange(n_t-1):
    Lb.append(Apows[i].dot(B))
    Apows.append(A.dot(Apows[-1]))
L = [Lb]
for i in xrange(n_t-1):
    L.append((i+1)*[np.zeros((n_x, n_u))] + Lb[:-(i+1)])
L = np.vstack(map(lambda L: np.hstack(L[::-1]), L[::-1]))
Apows = np.vstack(Apows)

# Initial condition and desired state trajectory
x0 = [1, 0]
Xr = np.array([[3, 0]] * n_t)

# Optimal control
effort_weight = 0.001
n_tu = n_t*n_u
y = Xr.flatten() - Apows.dot(x0)
LL = L.T.dot(L)
Ly = L.T.dot(y)
cost = lambda U: np.sum((L.dot(U) - y)**2) + effort_weight*np.sum(U**2)
cost_jac = lambda U: 2*(LL.dot(U) - Ly + effort_weight*U.T)
cost_hess = lambda U: 2*(LL + effort_weight*np.eye(n_tu))
opt = minimize(fun=cost, x0=np.zeros((n_tu, 1)), method="SLSQP", jac=cost_jac, hess=cost_hess, bounds=[u_lims]*n_tu)  # use Newton-CG if hess
print "Optimization Success: {}".format(opt.success)
U = opt.x.reshape((n_t, n_u))
# U = npl.pinv(L).dot(y).reshape((n_t, n_u))  # quadratic-cost, unconstrained simple analytical solution

# Simulation (iterative, as verification of L)
# K = np.array([[20, 10]]); U = []  # uncomment for naive hand-tuned controller
X = np.zeros((n_t, n_x))
for i, t in enumerate(T[1:]):
    # U.append(np.clip(K.dot(Xr[i] - X[i]), u_lims[0], u_lims[1]))  # uncomment for naive hand-tuned controller
    X[i+1] = A.dot(X[i]) + B.dot(np.clip(U[i], u_lims[0], u_lims[1]))
# U.append([0])  # uncomment for naive hand-tuned controller

# Visualization
fig, (ax0, ax1) = pyplot.subplots(2, 1)
fig.suptitle("Result", fontsize=16)
ax0.plot(T, X[:, 0], 'k', label="Position")
ax0.plot(T, X[:, 1], 'b', label="Velocity")
ax0.plot(T, Xr[:, 0], 'k--', label="Desired Position")
ax0.plot(T, Xr[:, 1], 'b--', label="Desired Velocity")
ax0.set_xlim(T[0], T[-1])
ax0.set_ylabel("State", fontsize=16)
ax0.grid(True)
ax0.legend(fontsize=16)
ax1.plot(T, U, 'r')
ax1.set_xlim(T[0], T[-1])
ax1.set_ylabel("Effort", fontsize=16)
ax1.set_xlabel("Time", fontsize=16)
ax1.grid(True)
pyplot.show()
