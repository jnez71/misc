#!/usr/bin/env python3
"""
Efficient implementation of a Bezier surface and its differential geometry.

"""
from __future__ import division
import numpy as np

################################################## CORE

class Bezier(object):
    """
    Bezier manifold of dimension 2 embedded in Euclidean space of dimension 3.

    """
    def __init__(self, knots=None):
        if knots is None:
            # Default to identity patch
            n = 4
            knots = np.zeros((n, n, 3), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    knots[i, j] = np.float64((i, j, 0)) / (n-1)
        self.set_knots(knots)

    def set_knots(self, knots):
        """
        Provide the control knots in an array with the first two
        dimensions indexing which knot and the third dimension
        holding the Euclidean coordinates of each knot.

        """
        self.knots = np.array(knots, dtype=np.float64)
        self.degree_x = self.knots.shape[0] - 1
        self.degree_y = self.knots.shape[1] - 1
        self.dimension = self.knots.shape[2] - 1
        assert self.degree_x > 0
        assert self.degree_y > 0
        assert self.dimension == 2
        self.dknots_x = self.degree_x * np.diff(self.knots, axis=0)
        self.dknots_y = self.degree_y * np.diff(self.knots, axis=1)

    def evaluate(self, x, y):
        """
        De Casteljau's algorithm is used to map the given surface coordinates
        (each from 0.0 to 1.0) to their corresponding location in Euclidean space.

        """
        lerps_x = np.zeros((self.degree_x+1, self.dimension+1), dtype=np.float64)
        for i in range(len(lerps_x)):
            lerps_y = self.knots[i].copy()
            for j in range(self.degree_y):
                for k in range(self.degree_y - j):
                    lerps_y[k] = (1.0-y)*lerps_y[k] + y*lerps_y[k+1]
            lerps_x[i] = lerps_y[0]
        for i in range(self.degree_x):
            for k in range(self.degree_x - i):
                lerps_x[k] = (1.0-x)*lerps_x[k] + x*lerps_x[k+1]
        return lerps_x[0]

    def jacobian(self, x, y):
        """
        Returns the 2by3 Jacobian matrix of the `evaluate` function
        at the given argument. The Grammian of this is the metric tensor.

        """
        return np.column_stack((Bezier(self.dknots_x).evaluate(x, y),
                                Bezier(self.dknots_y).evaluate(x, y)))

    def metric(self, x, y):
        """
        Returns the 2by2 metric tensor at the given surface coordinates.

        """
        J = self.jacobian(x, y)
        return J.T.dot(J)

    def orientation(self, x, y, q=0.0):
        """
        Returns a rotation matrix describing the orientation of the normal
        coordinates at [`x`, `y`] with yaw angle `q` in radians.

        """
        J = self.jacobian(x, y)
        rx, ry = (J / np.linalg.norm(J, axis=0)).T
        normal = np.cross(rx, ry)
        ncrossx = np.cross(normal, rx)  # must be re-unitized to mitigate roundoff error
        tangent = np.cos(q)*rx + np.sin(q)*(ncrossx / np.linalg.norm(ncrossx))
        binormal = np.cross(normal, tangent)
        R = np.column_stack((tangent, binormal, normal))
        return R / np.linalg.norm(R, axis=0)  # must be re-unitized to mitigate roundoff error

    def plot(self, n=40, block=True):
        """
        Plots this surface discretized by the given grid size `n`.
        Also shows the control knots and the central normal coordinate system.

        """
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        mesh = np.linspace(0.0, 1.0, n)
        points = np.transpose([self.evaluate(x, y) for x in mesh for y in mesh])
        quiver_origins = np.transpose([self.evaluate(mesh[n//2], mesh[n//2])]*3)
        quiver_arrows = self.orientation(mesh[n//2], mesh[n//2])
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("bezier", fontsize=12)
        ax.set_xlabel("rx", fontsize=12)
        ax.set_ylabel("ry", fontsize=12)
        ax.set_zlabel("rz", fontsize=12)
        ax.scatter(*self.knots.reshape(-1, 3).T, c='r', s=80)
        ax.scatter(*points, c=points[-1, :], s=60, marker='o', edgecolors=None)
        ax.quiver(quiver_origins[0], quiver_origins[1], quiver_origins[2],
                  quiver_arrows[0], quiver_arrows[1], quiver_arrows[2],
                  length=0.25, color=(1.0, 0.5, 0.0), lw=2.5)
        ax.axis("equal")
        pyplot.show(block=block)

################################################## TEST

if __name__ == "__main__":

    # Initialize a flat set of knots
    knots = np.zeros((5, 4, 3), dtype=np.float64)
    for i in range(knots.shape[0]):
        for j in range(knots.shape[1]):
            knots[i, j] = np.float64((i, j, 0))

    # Mess with the knots to make them more interesting
    knots[:, :, 0] *= -1.0
    knots[1:3, 1:3, 2] = -1.0
    knots[1:3, 0, 2] = (0.25, 0.5)
    knots[-1, -1, :] = (-4/2, 3/2, 0.5)

    # Construct the Bezier surface
    bezier = Bezier(knots)

    # Verify the analytical Jacobian against finite-differences at a random location
    x, y = np.random.sample(2)
    r = bezier.evaluate(x, y)
    d = 1e-6
    drdx = (bezier.evaluate(x+d,   y) - r) / d
    drdy = (bezier.evaluate(  x, y+d) - r) / d
    assert np.allclose(np.column_stack((drdx, drdy)), bezier.jacobian(x, y), atol=10*d)

    # Verify that the metric tensor computation is consistent with finite-differences
    assert np.allclose([[drdx.dot(drdx), drdx.dot(drdy)],
                        [drdy.dot(drdx), drdy.dot(drdy)]], bezier.metric(x, y), atol=10*d)

    # Verify that the orientation calculation returns an orthonormal matrix
    R = bezier.orientation(x, y, 2*np.pi*np.random.sample())
    assert np.allclose(R.dot(R.T), np.eye(3))

    # Plot the corresponding Bezier surface to visually inspect
    bezier.plot()
