"""
Feedforward neural network (NN) trained by extended kalman filter (EKF).

"""
# Dependencies
from __future__ import division
import time
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

# Function for neuron activation, and its recursive derivative
sig = lambda V: (1 + np.exp(-V))**-1
dsig = lambda sigV: sigV * (1 - sigV)

# Function for pushing signals through a synapse with bias
affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

# Function for pushing signals through a given NN
def NN(W, U, get_l=False):
    if U.ndim == 1: U = U[:, np.newaxis]
    l = sig(affine_dot(W[0], U))
    h = affine_dot(W[1], l)
    if get_l: return h, l
    return h

# Dimensionality of input, output, and hidden-layer
nu = 1
ny = 1
nl = 12

# Initial synapse weight matrices and weight covariance
W = [5*(2*np.random.sample((nl, nu+1))-1),
     5*(2*np.random.sample((ny, nl+1))-1)]
nW = sum(map(np.size, W))

# Training data
stdev = 0.05
U = np.arange(-10, 10, 0.1)
Y = np.exp(-U**2) + np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))
assert len(U) == len(Y)

# Process, sensor, and initial state covariances
Q = 0*np.eye(nW)
R = (stdev**2)*np.eye(ny)
P = 0.2*np.eye(nW)

# Runtime management
nepochs = 400
last_heartbeat = 0  # s
heart_rate = 1  # s

# Shuffle data between epochs
for epoch in xrange(nepochs):
    rand_idx = np.random.permutation(len(U))
    U_shuffled = U[rand_idx]
    Y_shuffled = Y[rand_idx]

    # Train
    for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):

        # Forward propagation
        h, l = NN(W, u, get_l=True)

        # Compute NN jacobian
        D = (W[1][:, :-1]*dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(ny, W[0].size),
                       spl.block_diag(*np.tile(np.concatenate((l, [1])), ny).reshape(ny, nl+1))))

        # Kalman gain
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(npl.inv(S))

        # Update weight estimates and covariance
        dW = K.dot(y-h)
        W[0] = W[0] + dW[:W[0].size].reshape(W[0].shape)
        W[1] = W[1] + dW[W[0].size:].reshape(W[1].shape)
        P = P - K.dot(H).dot(P) #+ Q  ### UNCOMMENT WHEN NONZERO

    # Heartbeat
    if time.time() - last_heartbeat > heart_rate:
        print("  Epoch: {}/{}\n------------------".format(epoch, nepochs))
        print("    MSE: {}".format(np.round(np.mean(np.square(Y - NN(W, U))), 6)))
        print("tr(Cov): {}\n------------------\n\n".format(np.round(np.trace(P), 6)))
        last_heartbeat = time.time()

# Evaluation
x = np.arange(-15, 15, 0.01)
f = NN(W, x)
plt.scatter(U, Y, c='r')
plt.plot(x, f, c='b', linewidth=3)
plt.grid(True)
plt.show()
