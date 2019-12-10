#!/usr/bin/env python3
"""
Solving a PDE that models an elastic solid.
https://en.wikipedia.org/wiki/Linear_elasticity
https://en.wikipedia.org/wiki/Finite_strain_theory

"""
import os; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
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

# Displacement and velocity initial condition
u = np.zeros((nx, ny, 2), float)
v = np.zeros((nx, ny, 2), float)

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

# Computes jacobian of the given vector field
def jac(u):
    ux_x, ux_y = np.gradient(u[:, :, 0])
    uy_x, uy_y = np.gradient(u[:, :, 1])
    return np.reshape(np.dstack((ux_x, ux_y, uy_x, uy_y)),
                      (u.shape[0], u.shape[1], u.shape[2], u.shape[2]))

# Computes transpose of the given 2-tensor field
def ttr(F):
    return np.transpose(F, (0, 1, 3, 2))

# Computes divergence of the given 2-tensor field
def div(F):
    Fxx_x, _ = np.gradient(F[:, :, 0, 0])
    _, Fxy_y = np.gradient(F[:, :, 0, 1])
    Fyx_x, _ = np.gradient(F[:, :, 1, 0])
    _, Fyy_y = np.gradient(F[:, :, 1, 1])
    return np.dstack((Fxx_x+Fxy_y, Fyx_x+Fyy_y))

################################################## GRAPHICS

# Configuration
window = (3*ny, 3*nx)
color_bg = (0, 0, 0, 255)  # background color
color_mg = (64, 64, 64, 128)  # midground color
color_fg = (255, 255, 255, 255)  # foreground color
res = 20  # grid resolution

# Initialize display
display = pygame.display.set_mode(window)
pygame.display.set_caption("JELLO BOI")

# Computes the pixel coordinate corresponding to the grid displacement
def pixel(u, i, j):
    return (ny + ny*(u[i, j, 1] + y[j])/ly, nx*(u[i, j, 0] + x[i])/lx)

# Inverse of the pixel function
def invpixel(px, py):
    return np.array((lx*py/nx - x[i], ly*(px - ny)/ny - y[j]), float)

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
            points_fg.append(pixel(u, i, j))
        pygame.draw.aalines(display, color_mg, False, points_mg)
        pygame.draw.aalines(display, color_fg, False, points_fg)
    # Y grid
    for j in range(0, ny, ny//res):
        points_mg = []
        points_fg = []
        for i in range(0, nx, nx//res):
            points_mg.append((ny+j, i))
            points_fg.append(pixel(u, i, j))
        pygame.draw.aalines(display, color_mg, False, points_mg)
        pygame.draw.aalines(display, color_fg, False, points_fg)
    # Refresh
    pygame.display.update()

################################################## SIMULATION

# Preparation
running = True
print("Running simulation!")
print("Click to interact. Press 'r' to reset.")
print("Close display window or ^C to quit.")

# Main loop
while running:
    # Record loop start time (in milliseconds)
    start_time = pygame.time.get_ticks()

    # Boundary conditions
    u[0, :] = 0.0
    U = jac(u)
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[-1, :] = 0.0

    # Handle user interface
    for event in pygame.event.get():
        # Quit
        if event.type == pygame.QUIT:
            running = False
            break
        # Reset
        elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_r):
            u = np.zeros_like(u)
            v = np.zeros_like(v)
            #U = np.zeros_like(U)  # leaving this out is a... feature
    # Mouse interaction
    if pygame.mouse.get_pressed()[0]:
        u[-5*nx//res:, -5*ny//res:] = invpixel(*pygame.mouse.get_pos()) + (3*lx/res, 3*ly/res)
        U[-5*nx//res:, -5*ny//res:] = 0.0
    # Update display
    show(u)

    # Green strain and Cauchy stress from linear elasticity
    E = (ttr(U) + U) / 2.0  # ??? can we handle the quadratic term, +U'U/2?
    S = K*E  # ??? need to use actual tensor field for shear stiffness

    # Acceleration from conservation of momentum
    a = g + (div(S) - c*v)/m

    # Temporal integration
    v += a*dt
    u += v*dt + 0.5*a*dt**2
    t += dt

    # Real time throttle
    remaining_time = int(1000*dt) - (pygame.time.get_ticks() - start_time)
    if remaining_time > 0:
        pygame.time.wait(remaining_time)
