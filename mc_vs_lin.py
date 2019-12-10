#!/usr/bin/env python3
"""
Demo of using monte-carlo vs linearization to approximate
a transformed random variable's probability distribution.
Automatic differentiation is used to find jacobians.
The "state" terminology comes from the field of stochastic processes.

"""
from autograd import numpy as np, jacobian  # https://github.com/HIPS/autograd
from matplotlib import pyplot
from matplotlib.patches import Ellipse

# Some wacky function that returns the next state given current state
# (state here is assumed to be two dimensional)
def process(x):
    # Arbitrary highly-nonlinear operations
    M = np.array([[x[0],  3.0],
                  [ 2.0, x[1]]])
    Minv = np.linalg.inv(M)
    dxdt = np.clip(np.dot(Minv, x) + [3, -3], -5, 5)
    return x + dxdt  # kinda looks like an integration scheme for no reason

# Some current state, Gaussian is fine
mean_now = np.array([0.1, -0.2])
covar_now = np.array([[1.0, 0.7],
                      [0.7, 0.8]])

# Monte Carlo to see how state actually transforms through process
n_particles = 5000  # reasonable choice for this number grows exponentially with state dimensionality
particles_now = np.random.multivariate_normal(mean_now, covar_now, n_particles)  # sample from current Gaussian
particles_next = np.array([process(x) for x in particles_now])  # unless process is linear, these won't be Gaussian distributed anymore

# Grab some statistics just for examination later
mc_mean_now = np.mean(particles_now, axis=0)
mc_covar_now = np.cov(particles_now, rowvar=False)
mc_mean_next = np.mean(particles_next, axis=0)
mc_covar_next = np.cov(particles_next, rowvar=False)

# Linearization to approximate state transforming as a Gaussian ("EKF Predict")
F = jacobian(process)(mean_now)  # mean_now is being used as the linearization point
ekf_mean_next = process(mean_now)
ekf_covar_next = F.dot(covar_now).dot(F.T)

# Quick verification that autograd computed F correctly- validate against finite-differences
def finite_difference(f, x, d=1e-8):
    zeros = np.zeros_like(x)
    fx = f(x)
    dfdx = []
    for i in range(len(x)):
        dx = zeros.copy()
        dx[i] = d
        dfdx.append(f(x+dx) - fx)
    return np.transpose(dfdx) / d
assert np.allclose(F, finite_difference(process, mean_now))

# Helper function for plotting covariance ellipses
def plot_covar(ax, covar, mean, conf=2*np.sqrt(6), edgecolor='k', linewidth=2, label=""):
    eigvals, eigvecs = np.linalg.eigh(covar)
    ellipse = Ellipse(xy=mean, width=conf*np.sqrt(eigvals[-1]), height=conf*np.sqrt(eigvals[0]),
                      angle=np.rad2deg(np.arctan2(eigvecs[1, -1], eigvecs[0, -1])),
                      facecolor='none', edgecolor=edgecolor, linewidth=linewidth, label=label)
    ax.add_artist(ellipse)

# Visualize
alpha = np.clip(1000/n_particles, 0.001, 1.0)
fig = pyplot.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(*particles_now.T, c='b', alpha=alpha, s=2, label="particles_now")
ax.scatter(*particles_next.T, c='r', alpha=alpha, s=2, label="particles_next")
ax.scatter(*mean_now, c='k', s=40, label="mean_now")
plot_covar(ax, covar_now, mean_now, edgecolor='k', label="covar_now")
ax.scatter(*mc_mean_now, c='g', s=40, label="mc_mean_now")
plot_covar(ax, mc_covar_now, mc_mean_now, edgecolor='g', label="mc_covar_now")
ax.scatter(*mc_mean_next, c='g', s=40, label="mc_mean_next")
ax.scatter(*ekf_mean_next, c='m', s=40, label="ekf_mean_next")
plot_covar(ax, mc_covar_next, mc_mean_next, edgecolor='g', label="mc_covar_next")
plot_covar(ax, ekf_covar_next, ekf_mean_next, edgecolor='m', label="ekf_covar_next")
ax.legend()
pyplot.show()
