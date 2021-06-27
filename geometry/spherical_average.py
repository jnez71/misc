#!/usr/bin/env python3
"""
We often have data that lives on a sphere and we want something like a sample average of it.
From a probability theory standpoint, the expected-value has no reason to live on the sphere
(the expected-value of a distribution can easily have low or zero probability itself).
However, we may still insist on computing a meaningful "average" that is indeed a point on
the sphere. It is intuitive to take the usual Euclidean sample average and then renormalize
to the sphere, but is this justified? A more general desire might be to select the point
on the manifold that minimizes the sum of squared geodesic distances to every sample.
This script demonstrates empirically that they are nearly equal in the case of a 2-sphere
where most of the data isn't antipodal (i.e. most samples are in one hemisphere).

"""
from autograd import numpy as np, value_and_grad  # pip3 install autograd
from scipy.optimize import minimize
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, sign=' ')
EPS = 1e-12

##################################################

def chord(u1, u2):
    return np.linalg.norm(u2 - u1, axis=u2.ndim-1)

def geodesic(u1, u2):
    # https://en.wikipedia.org/wiki/Great-circle_distance#From_chord_length
    return 2*np.arcsin(np.clip(chord(u1, u2) / 2, -1+EPS, 1-EPS))

##################################################

@value_and_grad
def leastsquares(u, samples, metric):
    return np.sum(np.square(metric(u, samples)))

def solve(samples, metric):
    return minimize(fun=leastsquares,
                    args=(samples, metric),
                    x0=[-1,1,0],
                    method="SLSQP",
                    jac=True,
                    constraints={"type": "eq",
                                 "fun": lambda u: np.linalg.norm(u)**2 - 1,
                                 "jac": lambda u: 2*u},
                    #options={"disp": True},
                    tol=EPS).x

##################################################

n = 500
samples = [1,0,0] + np.random.normal(0, 0.9, size=(n,3))
samples /= np.linalg.norm(samples, axis=1).reshape(n,-1)

min_chords = solve(samples, chord)
min_geodesics = solve(samples, geodesic)

renormalized = np.mean(samples, axis=0)
renormalized /= np.linalg.norm(renormalized)

##################################################

print("ArgMin of Chords:   ", min_chords)
print("ArgMin of Geodesics:", min_geodesics)
print("Renormalized Mean:  ", renormalized)

fig = pyplot.figure()
axis = fig.add_subplot(111, projection="3d")
axis.scatter(*samples.T, c='k', alpha=0.2, label="Samples")
axis.scatter(*min_chords.T, c='r', s=200, label="ArgMin of Chords")
axis.scatter(*min_geodesics.T, c='g', s=200, label="ArgMin of Geodesics")
axis.scatter(*renormalized.T, c='b', s=200, label="Renormalized Mean")
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_zlabel("z")
axis.legend()
pyplot.show()

##################################################
