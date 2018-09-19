#!/usr/bin/env python2
"""
Demo of computing a differentiable distance field for an arbitrary polygon.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from matplotlib import pyplot

viz_3d = False
if viz_3d: from mpl_toolkits.mplot3d import Axes3D

polygon = np.float64([[-1,  2],
                      [ 0,  4],
                      [ 3, -1],
                      [-1, -1]])
centroid = np.mean(polygon, axis=0)

lines = []
for i in xrange(len(polygon)):
    if i+1 < len(polygon):
        lines.append([polygon[i], polygon[i+1]])
    else:
        lines.append([polygon[i], polygon[0]])
lines = np.float64(lines)

def dist_from_line(point, line):
    direc = line[1] - line[0]
    length = npl.norm(direc)
    proj = np.dot(point-line[0], direc)*direc/length**2 + line[0]
    bounds = np.sort(line, axis=0)
    proj = np.clip(proj, bounds[0, :], bounds[1, :])
    return npl.norm(point - proj)

step = 0.1
xoffset, yoffset = np.max(np.abs(polygon), axis=0) + 1
extent = (centroid[0]-xoffset, centroid[0]+xoffset,
          centroid[1]-yoffset, centroid[1]+yoffset)
X = np.arange(extent[0], extent[1], step)
Y = np.arange(extent[2], extent[3], step)

F = np.zeros((len(X), len(Y)), dtype=np.float64)
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        F[i, j] = np.min([dist_from_line((x, y), line) for line in lines])
Xmesh, Ymesh = np.meshgrid(X, Y, indexing='ij')

fig = pyplot.figure()
if viz_3d:
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Xmesh, Ymesh, F, cmap="coolwarm", linewidth=0, antialiased=True)
else:
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(F.T, cmap='jet', interpolation='nearest', origin="lower", extent=extent)
for line in lines:
    ax.plot(line[:, 0], line[:, 1], c="w")
pyplot.show()
