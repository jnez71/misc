#!/usr/bin/env python3
"""
Interactively solving a PDE that models an elastic solid.
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
xy = np.dstack(np.meshgrid(x, y)[::-1])
t = 0.0

# Cardinality of discrete space and time
nx = len(x)
ny = len(y)
nt = np.inf

################################################## FIELDS

# State initial condition
u = None  # displacement
v = None  # time derivative
U = None  # spatial gradient
r = None  # material
def initialize():
    global u, v, U, r
    u = np.zeros((nx, ny, 2), float)
    v = np.zeros((nx, ny, 2), float)
    U = np.zeros((nx, ny, 2, 2), float)
    r = xy.copy()
initialize()

# Gravity
g = np.zeros((nx, ny, 2), float)
for i in range(nx):
    for j in range(ny):
        g[i, j] = (0.2, 0.0)

################################################## PROPERTIES

# Constitutive stiffness
K = 2e4  # scalar for simplicity, but restricts that tensile modulus == shear modulus

# Poisson's ratio
k = 0.0

# Mass density
m = 0.5

# Damping density
c = 0.1

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
    Fxx_x = np.gradient(F[:, :, 0, 0], axis=0)
    Fxy_y = np.gradient(F[:, :, 0, 1], axis=1)
    Fyx_x = np.gradient(F[:, :, 1, 0], axis=0)
    Fyy_y = np.gradient(F[:, :, 1, 1], axis=1)
    return np.dstack((Fxx_x+Fxy_y, Fyx_x+Fyy_y))

# Applies boundary conditions to the current state
def bound():
    global u, v, U, r
    # Fixed points
    u[0, :] = 0.0
    v[0, :] = 0.0

    ## Wall collisions
    #xlim = 0.0
    #ylim = -(ly/2.0)*(window_scale-1)
    #esc_x = np.where(r[:, :, 0] < xlim)
    #u[esc_x[0], esc_x[1], 0] = xlim - xy[esc_x[0], esc_x[1], 0] + dx
    #esc_y = np.where(r[:, :, 1] < ylim)
    #u[esc_y[0], esc_y[1], 1] = ylim - xy[esc_y[0], esc_y[1], 0] + dy

    # Compute spatial gradient
    U = jac(u)
    # Free surfaces
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[-1, :] = 0.0
    # Material coordinates
    r = u + xy

################################################## GRAPHICS

# Configuration
window_scale = 5
window = (window_scale*ny, window_scale*nx)
color_bg = (0, 0, 0, 255)  # background color
color_fg = (255, 255, 255, 255)  # foreground color
res = 20  # grid resolution
grab_size = 2

# Initialize display
display = pygame.display.set_mode(window)
pygame.display.set_caption("JELLO BOI")

# Computes the pixel coordinate corresponding to the material coordinate
def pixel(rij):
    return ((window[0]-ny)/2 + ny*rij[1]/ly,
            nx*rij[0]/lx)

# Inverse of the pixel function
def invpixel(pij):
    return np.array((lx*pij[1]/nx,
                     ly*(pij[0] - (window[0]-ny)/2)/ny), float)

# Visualizes a vector field on the global display
def show(u):
    global display
    # Clear
    display.fill(color_bg)
    # X grid
    for i in range(0, nx, nx//res):
        points = [pixel(r[i, j]) for j in range(0, ny, ny//res)]
        pygame.draw.aalines(display, color_fg, False, points)
    # Y grid
    for j in range(0, ny, ny//res):
        points = [pixel(r[i, j]) for i in range(0, nx, nx//res)]
        pygame.draw.aalines(display, color_fg, False, points)
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

    # Administrative interface
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Quit
            running = False
            break
        elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_r):
            # Reset
            initialize()

    # Interactions
    if pygame.mouse.get_pressed()[0]:
        # Mouse interaction drags rigid sub-block
        select = np.s_[-grab_size*nx//res:, -grab_size*ny//res:]
        u[select] = invpixel(pygame.mouse.get_pos()) - (lx, ly)
        v[select] = 0.0
    bound()

    # Update display
    show(u)

    # Green strain and Cauchy stress from linear elasticity
    E = (ttr(U) + U) / 2.0  # leaving out finite-strain quadratic term: +U'U/2
    S = K*(E + k*E[:, :, ::-1, ::-1])

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
