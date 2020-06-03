#!/usr/bin/env python2
"""
Creates various combinations of latitude and longitude lines
to demonstrate different grid types for globes. (This requires
the MayaVi visualization library).

The last scene crudely shows what SpaceX's satellite constellation
*may* look like. Altitudes are to scale, but satellite bodies are
exaggerated for visibility. Data was found here:
https://cdn3.vox-cdn.com/uploads/chorus_asset/file/8174403/SpaceX_Application_-.0.pdf

"""
from __future__ import division
import numpy as np; npl = np.linalg
import mayavi.mlab as maya

#####################################################################

def R(roll, pitch, yaw):
    """
    Generates the 3D rotation matrix that corresponds to the given Euler angles.
    The Euler angles are accepted in DEGREES. Sequence of application is extrinsic x-y-z.

    """
    ai, aj, ak = np.deg2rad((roll, pitch, yaw))
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk
    return np.array([[cj*ck, sj*sc-cs, sj*cc+ss],
                     [cj*sk, sj*ss+cc, sj*cs-sc],
                     [  -sj,    cj*si,    cj*ci]])

def grid_lines(form, n, R=np.eye(3), res=0.01):
    """
    Returns n evenly-spaced unit-sphere grid-lines with orientation R and plotting resolution res.
    The form is either 'lat' (latitude) or 'lon' (longitude).
    R is a 3D rotation matrix (use the R function above to generate it from Euler angles).
    The grid lines are returned as an array of shape (n, 3, 2*pi/res).
    Example: grid_lines(...)[3, 1, 4] is the third line's y-coordinate of its fourth point.

    """
    U = np.linspace(-np.pi/2, np.pi/2, n)
    V = np.arange(-np.pi, np.pi+res, res)
    lines = np.zeros((n, 3, len(V)))
    for i, u in enumerate(U):
        for j, v in enumerate(V):
            if form == 'lat': lines[i, :, j] = np.cos(u)*np.cos(v), np.cos(u)*np.sin(v), np.sin(u)
            elif form == 'lon': lines[i, :, j] = np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)
            else: raise ValueError("Invalid form. Choose 'lat' or 'lon'.")
        lines[i] = R.dot(lines[i])
    return lines

class Sphere(object):
    """
    Generates points for a sphere at the origin. Specify the radius and
    the resolution (number of steps from 0 to 2*pi) in the __init__.
    Call mysphere.draw(color) to add the sphere to the current maya figure.

    """
    def __init__(self, radius=1.0, res=40):
        self.u, self.v = np.mgrid[0:2*np.pi:res*1j, 0:np.pi:res*1j]
        self.x = radius*np.cos(self.u)*np.sin(self.v)
        self.y = radius*np.sin(self.u)*np.sin(self.v)
        self.z = radius*np.cos(self.v)
        self.draw = lambda color: maya.mesh(self.x, self.y, self.z, color=color)

#####################################################################

# Common parameters
n = 10  # number of grid_lines
lr = 0.01  # tube radius of the lines themselves
red, green, blue = map(tuple, np.eye(3))
black, gray, white = (0,)*3, (0.5,)*3, (1,)*3
sphere = Sphere()

fig1 = maya.figure(1, bgcolor=white, fgcolor=white)
for x, y, z in grid_lines('lat', n): maya.plot3d(x, y, z, color=red, tube_radius=lr)
for x, y, z in grid_lines('lon', n): maya.plot3d(x, y, z, color=blue, tube_radius=lr)
maya.title("lat-lon", color=black, height=0.875, size=0.5)
sphere.draw(gray)

fig2 = maya.figure(2, bgcolor=white, fgcolor=white)
for x, y, z in grid_lines('lon', n): maya.plot3d(x, y, z, color=red, tube_radius=lr)
for x, y, z in grid_lines('lon', n, R(90, 0, 0)): maya.plot3d(x, y, z, color=blue, tube_radius=lr)
maya.title("lon-lon", color=black, height=0.875, size=0.5)
sphere.draw(gray)

fig3 = maya.figure(3, bgcolor=white, fgcolor=white)
for x, y, z in grid_lines('lat', n): maya.plot3d(x, y, z, color=red, tube_radius=lr)
for x, y, z in grid_lines('lat', n, R(90, 0, 0)): maya.plot3d(x, y, z, color=blue, tube_radius=lr)
for x, y, z in grid_lines('lat', 3, R(0, 90, 0)): maya.plot3d(x, y, z, color=green, tube_radius=lr)
maya.title("lat-lat-orth", color=black, height=0.875, size=0.5)
sphere.draw(gray)

fig4 = maya.figure(4, bgcolor=white, fgcolor=white)
for x, y, z in grid_lines('lat', n): maya.plot3d(x, y, z, color=red, tube_radius=lr)
for x, y, z in grid_lines('lat', n, R(90, 0, 0)): maya.plot3d(x, y, z, color=blue, tube_radius=lr)
for x, y, z in grid_lines('lat', n-1, R(0, 90, 0)): maya.plot3d(x, y, z, color=green, tube_radius=lr)
maya.title("lat-lat-lat", color=black, height=0.875, size=0.5)
sphere.draw(gray)

fig5 = maya.figure(5, bgcolor=white, fgcolor=white)
for x, y, z in grid_lines('lat', n): maya.plot3d(x, y, z, color=black, tube_radius=lr/5)
for x, y, z in grid_lines('lon', n): maya.plot3d(x, y, z, color=black, tube_radius=lr/5)
for color, incl, dens, alt, num in zip([red, green, blue, (1, 0, 1)], [53, 74, 81, 70], [32, 8, 5, 6], [1110, 1130, 1275, 1325], [50, 50, 75, 75]):
    for i in np.linspace(0, 360, dens):
        for x, y, z in (1+alt/6371)*grid_lines('lon', 1, R(0, 90-incl, i)):
            maya.plot3d(x, y, z, color=color, tube_radius=lr/5)
            maya.points3d(x[::len(x)//num], y[::len(x)//num], z[::len(x)//num], scale_factor=0.02, color=color)
maya.title("SpaceX", color=black, height=0.875, size=0.5)
sphere.draw(gray)

maya.show()
