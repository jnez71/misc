#!/usr/bin/env python3
"""
Solving a PDE that models finite-strain elastic solids.

"""
import numpy as np  # pip3 install --user numpy
import pygame  # pip3 install --user pygame

################################################## DOMAIN

# Extent of space and time
lx = 1.0
ly = 1.0
lt = np.inf

# Discretization of space and time
dx = 0.005
dy = 0.005
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
K = 2e4

# Mass density
m = 0.5

# Damping density
c = 0.01

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

# Configuration
rate = 60  # updates per real second
color_bg = (0, 0, 0, 255)  # background color
color_mg = (64, 64, 64, 255)  # midground color
color_fg = (255, 255, 255, 255)  # foreground color
res = 10  # grid resolution

# Initialize display
display = pygame.display.set_mode((3*nx, 3*ny))
pygame.display.set_caption("JELLO BOI")

# Visualizes a vector field on the global display
def show(u):
    global display
    # Clear
    display.fill(color_bg)
    # X grid
    for i in range(0, nx, nx//res):
        points_mg = []
        points_fg = []
        for j in range(0, ny, ny//res):
            points_mg.append((ny+j, i))
            points_fg.append((ny+ny*(u[i, j, 1]+y[j])/ly, nx*(u[i, j, 0]+x[i])/lx))
        pygame.draw.aalines(display, color_mg, False, points_mg)
        pygame.draw.aalines(display, color_fg, False, points_fg)
    # Y grid
    for j in range(0, ny, ny//res):
        points_mg = []
        points_fg = []
        for i in range(0, nx, nx//res):
            points_mg.append((ny+j, i))
            points_fg.append((ny+ny*(u[i, j, 1]+y[j])/ly, nx*(u[i, j, 0]+x[i])/lx))
        pygame.draw.aalines(display, color_mg, False, points_mg)
        pygame.draw.aalines(display, color_fg, False, points_fg)
    # Refresh
    pygame.display.update()

################################################## SIMULATION

# Main loop
running = True
while running:
    start_time = pygame.time.get_ticks()

    # Boundary conditions
    u[0, :] = 0.0
    F = jac(u)
    F[:, 0] = 0.0
    F[:, -1] = 0.0
    F[-1, :] = 0.0

    # Green strain and Cauchy stress from linear elasticity
    E = (ttr(F) + F) / 2.0  # ??? try (FF-I)/2
    S = K*E  # ??? need to use actual tensor field

    # Acceleration from conservation of momentum
    a = g + (f - c*v + div(S))/m

    # Temporal integration
    v += a*dt
    u += v*dt + 0.5*a*dt**2
    t += dt

    # User interface
    show(u)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    # Real time throttle
    remaining_time = (1000//rate) - (pygame.time.get_ticks() - start_time)
    if remaining_time > 0:
        pygame.time.wait(remaining_time)
