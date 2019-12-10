#!/usr/bin/env python3
"""
Computational demo of the exponential map for the quaternion representation of SO3.
Quaternions here are stored as arrays [w, i, j, k]. (NOT the ROS TF convention).

"""
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

npl = np.linalg
PI = np.pi

################################################## FUNCTIONS

def exp(v):
    """
    Returns the unit-quaternion corresponding to a rotation-vector v.

    """
    mag = npl.norm(v)
    if np.isclose(mag, 0): return np.array([1, 0, 0, 0])
    angle_by_2 = np.mod(mag, 2*PI) / 2
    axis = np.divide(v, mag)
    return np.insert(np.sin(angle_by_2)*axis, 0, np.cos(angle_by_2))


def log(q):
    """
    Returns a rotation-vector corresponding to a unit-quaternion q.

    """
    vnorm = npl.norm(q[1:])
    if np.isclose(vnorm, 0): return np.zeros(3)
    if q[0] < 0: q = np.negative(q)  # optional convention
    qnorm = npl.norm(q)
    return 2*np.arccos(q[0]/qnorm)*(q[1:]/vnorm)


def compose(ql, qr):
    """
    Returns the product of two quaternions as ql*qr.

    """
    return np.array([-ql[1]*qr[1] - ql[2]*qr[2] - ql[3]*qr[3] + ql[0]*qr[0],
                      ql[1]*qr[0] + ql[2]*qr[3] - ql[3]*qr[2] + ql[0]*qr[1],
                     -ql[1]*qr[3] + ql[2]*qr[0] + ql[3]*qr[1] + ql[0]*qr[2],
                      ql[1]*qr[2] - ql[2]*qr[1] + ql[3]*qr[0] + ql[0]*qr[3]])


def invert(q):
    """
    Returns the inverse of a unit-quaternion q.

    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def act(q, v):
    """
    Returns the transformation (rotation) of a vector v by a unit-quaternion q.

    """
    return compose(compose(q, np.insert(v, 0, 0)), invert(q))[1:]

################################################## TESTS

# Test exp
assert np.allclose(exp([0, 0, PI/2]),
                   [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])  # 90deg rotation about z axis
assert np.allclose(exp([PI, 0, 0]),
                   [0, 1, 0, 0])  # 180deg rotation about x axis

# Test log
random_quat = np.random.sample(4)
random_quat /= npl.norm(random_quat)
assert np.allclose(exp(log(random_quat)), random_quat)  # exp and log are inverses

################################################## DEMO

# Arbitrary mean
q0 = [1, 0, 0, 0]

# Arbitrary covariance
P = [[0.01,   0,   0],
     [   0, 0.1,   0],
     [   0,   0, 0.5]]  # rad^2

# Generate samples
samples = []
for i in range(3000):

    # Sample a random tangent-vector to SO3
    v = np.random.multivariate_normal(np.zeros(3), P)  # Gaussian is arbitrary choice

    # Perturb mean by this tangent
    sample = compose(exp(v), q0)
    samples.append(sample)

    # ^ In the above, if q0 converts B-coordinates to A-coordinates, then composing exp(v)
    # on the left means the noise is in A-coordinates, while composing exp(v) on the right
    # would mean that the noise is in B-coordinates.

# Transform some point by the SO3 samples to demonstrate the distribution
point0 = [1, 0, 0]
points_x, points_y, points_z = zip(*[act(sample, point0) for sample in samples])

# Visualize distribution
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_x, points_y, points_z, c='b', alpha=0.075)
ax.set_title("Samples", fontsize=22)
ax.set_xlabel("x", fontsize=22)
ax.set_ylabel("y", fontsize=22)
ax.set_zlabel("z", fontsize=22)
pyplot.show()
