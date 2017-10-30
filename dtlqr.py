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
A = np.eye(n_x) + np.array([[ 0,  1],
                            [-1, -1]])*dt

# B = df/du
n_u = 1
B = np.array([[0],
              [1]])*dt

# Control limits
u_lims = (-10, 10)

# Efficiently compute the matrix powers of A and the simulation matrix L
# x(t) = L*u(t) + Apows*x(0),  L is lower triangular (system is Markovian)
Lb = [np.zeros((n_x, n_u))]
Apows = [np.eye(n_x)]
for i in xrange(n_t-1):
    Lb.append(Apows[i].dot(B))
    Apows.append(A.dot(Apows[-1]))
L = [Lb]
for i in xrange(n_t-1):
    L.append((i+1)*[np.zeros((n_x, n_u))] + Lb[:-(i+1)])
L = np.vstack(map(lambda L: np.hstack(L[::-1]), L[::-1]))
Apows = np.vstack(Apows)

# Desired state trajectory and initial condition
# Xr = np.array([[3, 0]] * n_t)  # constant waypoint
# x0 = [0, 0]
per = 3; amp = 3
Xr = np.vstack((amp*np.cos(2*np.pi/per * T), -amp*2*np.pi/per*np.sin(2*np.pi/per * T))).T  # track sinusoid
x0 = Xr[0]

# Optimal control
final_weight = 100
effort_weight = 0.001
y = Xr.flatten() - Apows.dot(x0)
LL = L.T.dot(L); LLf = L[-n_x:].T.dot(L[-n_x:])
yL = y.T.dot(L); yLf = y[-n_x:].T.dot(L[-n_x:])
def tcost(U):
    E = L.dot(U) - y
    return np.sum(E**2) + final_weight*np.sum(E[-n_x:]**2) + effort_weight*np.sum(U**2)
def tcost_jac(U):
    return 2*(U.T.dot(LL + final_weight*LLf) - (yL + final_weight*yLf) + effort_weight*U.T)
opt = minimize(fun=tcost, jac=tcost_jac, x0=np.zeros((n_t*n_u, 1)), method="SLSQP", bounds=[u_lims]*n_t)
print "Optimization Success: {}".format(opt.success)
U = opt.x.reshape((n_t, n_u))

# Simulation (iterative, as verification of L)
X = np.zeros((n_t, n_x)); X[0] = x0
U_act = [np.clip(U[0], u_lims[0], u_lims[1])]
for i, t in enumerate(T[1:]):
    U_act.append(np.clip(U[i+1], u_lims[0], u_lims[1]))
    X[i+1] = A.dot(X[i]) + B.dot(U_act[i])

# Visualization
fig, (ax0, ax1) = pyplot.subplots(2, 1)
fig.suptitle("Results", fontsize=16)
ax0.plot(T, X[:, 0], 'k', label="Position")
ax0.plot(T, X[:, 1], 'b', label="Velocity")
ax0.plot(T, Xr[:, 0], 'k--', label="Position Desired")
ax0.plot(T, Xr[:, 1], 'b--', label="Velocity Desired")
ax0.set_xlim(T[0], T[-1])
ax0.set_ylabel("State", fontsize=16)
ax0.grid(True)
ax0.legend(fontsize=10)
ax1.plot(T, U_act, 'r', label="Effort Achieved")
ax1.plot(T, U, 'r--', label="Effort Requested")
ax1.plot(T, [u_lims[0]]*n_t, 'y--', label="Actuator Limits")
ax1.plot(T, [u_lims[1]]*n_t, 'y--')
ax1.set_xlim(T[0], T[-1])
ax1.set_ylabel("Effort", fontsize=16)
ax1.set_xlabel("Time", fontsize=16)
ax1.grid(True)
ax1.legend(fontsize=10)
pyplot.show()
