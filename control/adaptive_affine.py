#!/usr/bin/env python3
"""
Adaptive Affine Control:
My favorite myopic (not MPC, DP, or RL) control-law when absolutely nothing is known about your system except
that the control is additive and fully-actuated:
```
dx/dt = f(x,t) + u         # drift f unknown, state x at time t known, choose control u to make x=r
u = W.dot(x) + b           # policy is affine function of state
dW/dt = outer(k*(r-x), x)  # parameters are adapted (learned online) to oppose the...
db/dt = k*(r-x)            # ... gradient of the error-energy-rate d/dt((k/2)*(r-x)^2)
```
Try this with any crazy f. Even throw in a B(x,t) transformation on u (though no guarantees for that).
It's basically PID but with the PD gains evolving according to the regression-like dW/dt I gave.
PID with stationary PD gains fails when the f is reasonably nonlinear. This law still works.
Of course, additive-control fully-actuated systems pretty much only model lame low-level problems, but still neat.

"""
# Dependencies
import numpy as np
from matplotlib import pyplot

##################################################

# Controller
class C:
    def __init__(self, n, k):
        self.n = int(n)
        self.k = float(k)
        self.W = np.zeros((n, n), dtype=float)
        self.b = np.zeros(n, dtype=float)
    def u(self, r, x, dt):
        ked = self.k*(r - x)*dt
        self.W += np.outer(ked, x)
        self.b += ked
        return self.W.dot(x) + self.b

##################################################

# Drift dynamic
n = 3
def f(x, t):
    return np.array([10.0*(x[1] - x[0]),
                     x[0]*(28.0 - x[2]) - x[1],
                     x[0]*x[1] - 2.6*x[2]])

# Actuator dynamic
# (needs to be identity for Lyapunov proof, but might still work otherwise)
def B(x, t):
    return np.array([[x[1],    0.0, 0.0],
                     [ 0.0, 2*x[0], 0.0],
                     [ 0.0,    0.0, 1.0]])

##################################################

# Time
dt = 0.001
T = np.arange(0.0, 3.0, dt)

# State
X = np.zeros((len(T), n), dtype=float)
X[0] = [-1.0, 2.0, 3.0]

# Control
U = np.zeros((len(T), n), dtype=float)
c = C(n, 1.0)

# Reference
R = np.array([[6.0, 7.0, -7.0]] * len(T))

##################################################

# Simulation
control = True
for i in range(len(T)-1):
    if control: U[i] = c.u(R[i], X[i], dt)
    dxdt = f(X[i], T[i]) + B(X[i], T[i]).dot(U[i])
    X[i+1] = X[i] + dxdt*dt

##################################################

# Plot
fig = pyplot.figure()
if control: fig.suptitle("Controlled Response", fontsize=26)
else: fig.suptitle("Natural Response", fontsize=26)
ax = None
for i in range(n):
    ax = fig.add_subplot(n, 1, i+1, sharex=ax)
    ax.plot(T, X[:, i], color='b', linewidth=2, label="state")
    ax.plot(T, R[:, i], color='g', linewidth=3, linestyle=':', label="desire")
    ax.plot(T[:-1], U[:-1, i], color='r', linewidth=0.5, label="action", scaley=False)
    ax.set_xlim([T[0], T[-1]])
    ax.set_ylabel("state "+str(i), fontsize=20)
    ax.grid(True)
ax.set_xlabel("time", fontsize=20)
ax.legend()
pyplot.show()
