#!/usr/bin/env python3
"""
Kernel of Gaussian-transition scalar Markov process?

"""
import numpy as np
from matplotlib import pyplot

npr = np.random
np.set_printoptions(suppress=True)

pyplot.rcParams["font.size"] = 16
pyplot.rcParams["axes.grid"] = True

################################################## SYSTEM

def initial(m=10.0, s=2.0):
    return npr.normal(m, s)  # gaussian initial-condition

def transition(x, s=1.0):
    #f = 0.5*x  # linear
    f = 10*np.sin(2/(1+x**2))  # nonlinear
    return f + npr.normal(0.0, s)  # gaussian transition

def simulate(d):
    X = [initial()]
    for i in range(d-1):
        X.append(transition(X[-1]))
    return X  # one sample from d-dimensional joint (only gaussian if linear transitions)

################################################## SIMULATE

d = 9
n = int(5e5)

print("Simulating samples...")
samples = np.array([simulate(d) for i in range(n)])

print("Computing statistics...")
mean = np.mean(samples, axis=0)
covar = np.cov(samples, rowvar=False)

################################################## VISUALIZE

print("========================================")
print(np.round(mean, 3), '\n')
print(np.round(covar, 3))
print("========================================")

print("Visualizing covariance...")
vmax = np.max(np.abs(covar))
pyplot.imshow(covar, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="lanczos")
pyplot.colorbar()
pyplot.grid(False)
pyplot.title("Covariance")

print("Visualizing joint...")
pyplot.figure()
pyplot.scatter(samples[::int(n/1e3+1), 0], samples[::int(n/1e3+1), -1], alpha=0.4)
pyplot.xlabel("x0")
pyplot.ylabel("x{0}".format(d-1))
pyplot.show()
