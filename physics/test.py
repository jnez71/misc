#!/usr/bin/env python3
"""
Numerical deformation gradients?

"""
import numpy as np
from matplotlib import pyplot

I = np.identity(2, float)

lx = 1.0
ly = 1.0

dx = 0.1
dy = 0.1

x = np.arange(0.0, lx+dx, dx, float)
y = np.arange(0.0, ly+dy, dy, float)
xy = np.dstack(np.meshgrid(y, x)[::-1])

nx = len(x)
ny = len(y)
nxy = (nx, ny)

a = np.deg2rad(45)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], float)
assert np.allclose(R.T.dot(R), I)

o = np.array([lx/2, ly/2], float)
r = np.einsum('ij,xyj->xyi', R, xy - o) + o

u = r - xy

def d_dxy(u):
    ux_x, ux_y = np.gradient(u[:, :, 0], dx, dy)
    uy_x, uy_y = np.gradient(u[:, :, 1], dx, dy)
    return np.reshape(np.dstack((ux_x, ux_y, uy_x, uy_y)),
                      (u.shape[0], u.shape[1], u.shape[2], u.shape[2]))

U = d_dxy(u)
F = U + I
assert np.allclose(F, R)

C = np.einsum("xyji,xyjk->xyik", F, F)
E = (C - I) / 2.0
assert np.allclose(E, 0.0)

pyplot.quiver(xy[:, :, 0], xy[:, :, 1], u[:, :, 0], u[:, :, 1], scale=8, scale_units='xy')
pyplot.axis('equal')
pyplot.grid(True)
pyplot.show()
