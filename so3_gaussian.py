#!/usr/bin/env python2
"""
Sampling from a normal distribution over SO3.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.linalg import expm

# Mean SO3 member
R0 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

# Covariance matrix is basically for roll-pitch-yaw,
# need only be positive-definite symmetric
S = 0.01*np.array([[10, 00, 00],
                   [00, 10, 00],
                   [00, 00, 30]])

# Lie algebra generator
crossmat = lambda v: np.array([[  0   , -v[2]  ,  v[1] ], 
                               [ v[2] ,   0    , -v[0] ], 
                               [-v[1] ,  v[0]  ,   0   ]])

# Generate samples
samples = []
for i in xrange(3000):

    # Sample a random normal (tangent) vector to SO3
    v = np.random.multivariate_normal(np.zeros(3), S)

    # Bijection to lie algebra
    V = crossmat(v)

    # Perturb mean
    samples.append(expm(V).dot(R0))

# Transform some point by the SO3 samples to demonstate spread
p0 = np.array([1, 0, 0])
points_x, points_y, points_z = zip(*[sample.dot(p0) for sample in samples])
c = R0.dot(p0)

# Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_x, points_y, points_z, c='b', alpha=0.1)
ax.scatter(c[0], c[1], c[2], c='r', alpha=1, s=50)
fig.suptitle("Samples", fontsize=22)
ax.set_xlabel("x", fontsize=22)
ax.set_ylabel("y", fontsize=22)
ax.set_zlabel("z", fontsize=22)
print("Full-screen it!")
plt.show()
