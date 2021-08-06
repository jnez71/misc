#!/usr/bin/env python3
"""
Demonstration that the Principle of Stationary Action is
*not* a principle of "least" action, even for simple systems
like a classical harmonic oscillator.

In this case, if you increase the time duration `b` passed PI,
the action becomes nonconvex and the minimization approach fails.

See:
    https://physics.stackexchange.com/q/122486/155516
    https://physics.stackexchange.com/a/122511/155516
    https://en.wikipedia.org/wiki/Conjugate_points

Philosophical note:
If you were once reassured by the notion that "nature extremizes"
or "solves an optimization problem", then perhaps you may be
swayed by the more precise notion that "nature balances",
"is indifferent", and "has fundamental symmetries".

"""
from autograd import numpy as np, value_and_grad  # pip3 install --user autograd
from scipy.optimize import minimize               # pip3 install --user scipy
from matplotlib import pyplot                     # pip3 install --user matplotlib

np.set_printoptions(suppress=True)
pyplot.rcParams["font.size"] = 16
pyplot.rcParams["axes.grid"] = True

##################################################

n = 1  # configuration-space dimensionality

m = 1.0  # mass
k = 1.0  # stiffness

dt = 0.01  # temporal resolution
b = np.pi#+0.2  # temporal extent (here PI is the value beyond which the action is nonconvex)

q0 = 1.0  # initial configuration
qd0 = 0.0  # initial velocity

#########################

def lagrangian(q, qd):
    return 0.5*m*qd**2 - 0.5*k*q**2  # kinetic less potential

def acceleration(q, qd):
    return -k*q/m  # force over mass

##################################################

def stationary(q0, qd0, dt, b):
    T = np.arange(0, b, dt)
    Q = np.empty((len(T), n), float)
    Qd = np.empty((len(T), n), float)

    Q[0] = q0
    Qd[0] = qd0

    for i in range(len(T)-1):
        # Verlet integration
        a = acceleration(Q[i], Qd[i])
        Q[i+1] = Q[i] + Qd[i]*dt + 0.5*a*dt**2
        a1 = acceleration(Q[i+1], Qd[i])
        Qd[i+1] = Qd[i] + 0.5*(a+a1)*dt

    return (T, Q, Qd)

print("Solving EL for stationary trajectory...")
T, Q, Qd = stationary(q0, qd0, dt, b)
qb = Q[-1]

##################################################

@value_and_grad
def action(Q, dt=dt):
    Qd = np.append(np.diff(Q)/dt, 0)
    return np.sum(lagrangian(Q, Qd)) * dt

boundaries = {
    "type": "eq",
    "fun": lambda Q: (Q[0]-q0)**2 + (Q[-1]-qb)**2
}

print("Directly optimizing for minimal trajectory...")
Qmin = minimize(action, np.zeros_like(Q), method="SLSQP",
                jac=True, constraints=boundaries,
                options={"disp": True, "maxiter": 1000}).x

##################################################

print("Plotting results...")
figure, axes = pyplot.subplots(1, 1)
axes.set_ylabel("Configuration")
axes.set_xlabel("Time")
axes.plot(T, Q, label="stationary (physical)")
axes.plot(T, Qmin, ls=":", lw=4, label="minimum")
axes.legend()
pyplot.show()

##################################################
