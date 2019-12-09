#!/usr/bin/env python3
"""
Solving a PDE that models finite-strain elastic solids.

"""
import numpy as np
import pygame

################################################## DOMAIN

# Extent of space and time
lx = 1.0
ly = 1.0
lt = np.inf

# Discretization of space and time
dx = 0.01
dy = 0.01
dt = 0.005

# Space and time
x = np.arange(0.0, lx, dx, float)
y = np.arange(0.0, ly, dy, float)
t = 0.0

# Cardinality of discrete space and time
nx = len(x)
ny = len(y)
nt = np.inf

################################################## FIELDS

# Deformation and velocity initial condition
u = np.zeros((nx, ny, 2), float)
v = np.zeros((nx, ny, 2), float)

# User force initial condition
f = np.zeros((nx, ny, 2), float)

# Gravity
g = np.zeros((nx, ny, 2), float)
for i in range(nx):
    for j in range(ny):
        g[i, j] = (1.0, 0.0)

################################################## PROPERTIES

# Constitutive stiffness
K = 1e4

# Mass density
m = 1.2

# Damping density
c = 1.0

################################################## OPERATORS

# Jacobian of a vector field
def jac(u):
    ux_x, ux_y = np.gradient(u[:, :, 0])
    uy_x, uy_y = np.gradient(u[:, :, 1])
    return np.reshape(np.dstack((ux_x, ux_y, uy_x, uy_y)),
                      (u.shape[0], u.shape[1], u.shape[2], u.shape[2]))

# Transpose of a tensor field
def ttr(F):
    return np.transpose(F, (0, 1, 3, 2))

# Divergence of a tensor field
def div(F):
    Fxx_x, _ = np.gradient(F[:, :, 0, 0])
    _, Fxy_y = np.gradient(F[:, :, 0, 1])
    Fyx_x, _ = np.gradient(F[:, :, 1, 0])
    _, Fyy_y = np.gradient(F[:, :, 1, 1])
    return np.dstack((Fxx_x+Fxy_y, Fyx_x+Fyy_y))

################################################## GRAPHICS



################################################## SIMULATION

np.set_printoptions(precision=4, suppress=True)

# Main loop
running = True
while running:

    # Boundary conditions
    u[0, :] = 0.0
    F = jac(u)
    F[:, 0] = 0.0
    F[:, -1] = 0.0
    F[-1, :] = 0.0

    # Green strain and Cauchy stress from linear elasticity
    E = (ttr(F) + F) / 2.0  # eventually try (FF-I)/2
    S = K*E

    # Acceleration from conservation of momentum
    a = g + (f - c*v + div(S))/m

    # Temporal integration
    v += a*dt
    u += v*dt + 0.5*a*dt**2
    t += dt


    print(u[nx//2, ny//2])
