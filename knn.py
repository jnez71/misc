"""
Feedforward neural network (NN) trained by extended kalman filter (EKF).

"""
# Dependencies
from __future__ import division
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

# Function for neuron activation and its recursive derivative
sig = lambda v: (1 + np.exp(-v))**-1
dsig = lambda sigv: sigv * (1 - sigv)

# Function for pushing a signal through a synapse with bias
affine_dot = lambda A, v: np.dot(A[:, :-1], v) + A[:, -1]

# Dimensionality of input, hidden-layer, and output
nu = 1
nl = 10
ny = 1

# Initial synapse weight matrices and weight covariance
np.random.seed(1)
W = [5*(2*np.random.sample((nl, nu+1))-1),
     5*(2*np.random.sample((ny, nl+1))-1)]
nW = W[0].size + W[1].size
P = 0.2*np.eye(nW)

# Training data
U = np.array([[4]])

# Training
for u in U:

    # Forward propagation
    l = sig(affine_dot(W[0], u))
    h = affine_dot(W[1], l)

    # Compute NN jacobian
    D = (W[1][:, :-1]*dsig(l)).flatten()
    H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(ny, W[0].size),
                   spl.block_diag(*np.tile(np.concatenate((l, [1])), ny).reshape(ny, nl+1))))

# Evaluation
pass
