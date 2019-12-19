#!/usr/bin/env python3
"""
Interactively solving a PDE that models an elastic solid.
https://en.wikipedia.org/wiki/Linear_elasticity
https://en.wikipedia.org/wiki/Finite_strain_theory

DEPENDENCIES:
sudo apt install python3 python3-pip
pip3 install --user numpy pygame

"""
# Math
import numpy as np

# Visualization
import os; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
try:
    # For mobile devices
    import pygame_sdl2 as pygame
    pygame.import_as_pygame()
    mobile = True
except ImportError:
    # For desktop devices
    import pygame
    mobile = False

################################################## DOMAIN

# Extent of space and time
lx = 1.0
ly = 1.0
lt = np.inf

# Discretization of space and time
dx = 0.015
dy = 0.015
dt = 0.0002

# Space and time
x = np.arange(0.0, lx, dx, float)
y = np.arange(0.0, ly, dy, float)
xy = np.dstack(np.meshgrid(y, x)[::-1])
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
g = 0.0
f = np.zeros((nx, ny, 2), float)
for i in range(nx):
    for j in range(ny):
        f[i, j] = (1.0, 0.0)

################################################## PROPERTIES

# Young's modulus (stiffness)
s = 1e6

# Poisson's ratio (contractivity)
k = 0.15

# Mass density
m = 0.1

# Damping density
c = 4.0

################################################## OPERATORS

# Computes the jacobian of the given vector field
def jacobian(u):
    ux_x, ux_y = np.gradient(u[:, :, 0])
    uy_x, uy_y = np.gradient(u[:, :, 1])
    return np.reshape(np.dstack((ux_x, ux_y, uy_x, uy_y)),
                      (u.shape[0], u.shape[1], u.shape[2], u.shape[2]))

# Computes the divergence of the given 2-tensor field
def divergence(F):
    Fxx_x = np.gradient(F[:, :, 0, 0], axis=0)
    Fxy_y = np.gradient(F[:, :, 0, 1], axis=1)
    Fyx_x = np.gradient(F[:, :, 1, 0], axis=0)
    Fyy_y = np.gradient(F[:, :, 1, 1], axis=1)
    return np.dstack((Fxx_x+Fxy_y, Fyx_x+Fyy_y))

# Returns the transpose of the given 2-tensor field
def transpose(F):
    return np.transpose(F, (0, 1, 3, 2))

# Computes the trace of the given 2-tensor field
def trace(F):
    return np.trace(F, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis]

# Applies boundary conditions to the current state
def bound():
    global u, v, U, r
    # Fixed points
    u[0, :] = 0.0
    v[0, :] = 0.0
    # Corner cases
    u[-2:, 0] = (u[-3, 0] + u[-1, 2]) / 2.0
    u[-2:, -1] = (u[-3, -1] + u[-1, -3]) / 2.0
    v[-2:, 0] = (v[-3, 0] + v[-1, 2]) / 2.0
    v[-2:, -1] = (v[-3, -1] + v[-1, -3]) / 2.0
    # Compute spatial gradient
    U = jacobian(u)
    # Free surfaces
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[-1, :] = 0.0
    # Material coordinates
    r = u + xy

# The judge decides the score
np.random.seed(0)
morals = np.random.uniform(-1.0, 1.0, v.size)
def judge():
    return morals.dot((v - np.mean(v)).flatten()) > 500

################################################## GRAPHICS

# Configuration
rate = 60  # updates per real second
mat = (300*nx/ny, 300)  # rectangular shape of material
env = (1000, 1500)  # rectangular shape of ambient environment
res = 16  # material grid resolution
xgrab, ygrab = 4, 4  # shape of sub-block grabbable by user
pretty = True  # whether or not to use pretty colors
fontsize = 25  # size of font in pixels for drawn text

# Initialize display
display = pygame.display.set_mode(env[::-1])
pygame.display.set_caption("JELLO BOI")
pygame.font.init()
font = pygame.font.SysFont("freemono", fontsize)

# Computes the pixel coordinate corresponding to the material coordinate
def pixel(rij):
    return ((env[1]-mat[1])/2 + mat[1]*rij[1]/ly,
            mat[0]*rij[0]/lx)

# Inverse of the pixel function
def invpixel(pij):
    return np.array((lx*pij[1]/mat[0],
                     ly*(pij[0] - (env[1]-mat[1])/2)/mat[1]), float)

# Computes the color associated with the given intensity scale
def scolor(scale):
    if pretty:
        return tuple(255*np.clip((scale, 1.0-scale**2, 1.0-scale, 1.0), 0.0, 1.0))
    else:
        return (255, 255, 255, 255)

# Visualizes a vector field on the global display
def show(u):
    global display
    # Clear
    display.fill((0, 0, 0, 255))
    # Horizontal grid
    for i in range(0, nx, nx//res):
        points = [pixel(r[i, j]) for j in range(0, ny, ny//res)]
        pygame.draw.aalines(display, scolor(np.linalg.norm(u[i, :])/(0.25*ny)), False, points)
    # Vertical grid
    for j in range(0, ny, ny//res):
        points = [pixel(r[i, j]) for i in range(0, nx, nx//res)]
        pygame.draw.aalines(display, scolor(np.linalg.norm(u[:, j])/(0.25*nx)), False, points)
    # Textual information
    image = font.render("    score: {}".format(score), True, (255, 255, 0, 255))
    display.blit(image, (10, 0))
    for i, text in enumerate(("stiffness: {}".format(s/1e3),
                              "squeeeesh: {}".format(k),
                              "  inertia: {}".format(m),
                              "  damping: {}".format(c),
                              "  gravity: {}".format(g/1e3),
                              "   v_grab: {}".format(xgrab-1),
                              "   h_grab: {}".format(ygrab-1))):
        if i == menu:
            color = (0, 255, 0, 255)
        else:
            color = (200, 200, 200, 200)
        image = font.render(text, True, color)
        display.blit(image, (10, (i+1)*(fontsize+5)))
    # Refresh
    pygame.display.update()

################################################## SIMULATION

# Preparation
print("Running simulation!")
print("-------------------")
print("Click: grab")
print("Arrow: menu")
print("    q: quit")
print("    r: reset")
print("    c: color")
print("    f: fullscreen")
print("-------------------")
score = 0
menu = 0
running = True

# Main loop
while running:
    # Record loop start time (in milliseconds)
    start_time = pygame.time.get_ticks()

    # Administrative interface
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Quit
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            # Button presses
            if event.key == pygame.K_q:
                # Quit
                running = False
            elif event.key == pygame.K_r:
                # Reset
                initialize()
            elif event.key == pygame.K_f:
                # Toggle full-screen
                pygame.display.toggle_fullscreen()
            elif event.key == pygame.K_c:
                # Toggle colors
                pretty = not pretty
            elif event.key in (pygame.K_UP, pygame.K_w):
                # Ascend menu
                menu = (menu - 1) % 7
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                # Descend menu
                menu = (menu + 1) % 7
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                # Increase property
                action = 1
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                # Decrease property
                action = -1
    # Apply menu action
    if menu == 0:
        s = np.round(np.clip(s + action*5e4, 0.0, 2e6), 6)
    elif menu == 1:
        k = np.round(np.clip(k + action*0.05, -0.4, 0.4), 6)
    elif menu == 2:
        m = np.round(np.clip(m + action*0.1, 0.05, 10.0), 6)
    elif menu == 3:
        c = np.round(np.clip(c + action*0.1, 0.0, 10.0), 6)
    elif menu == 4:
        g = np.round(np.clip(g + action*1e3, -3e3, 1e4), 6)
    elif menu == 5:
        xgrab = int(np.clip(xgrab + action, 0, res))
    elif menu == 6:
        ygrab = int(np.clip(ygrab + action, 0, res))

    # Interactions
    if pygame.mouse.get_pressed()[0]:
        # Mouse interaction drags rigid sub-block
        select = np.s_[-xgrab*nx//res:, -ygrab*ny//res:]
        u[select] = invpixel(pygame.mouse.get_pos()) - (lx, ly) + (lx/res, ly/res)
        v[select] = 0.0
    bound()

    # Update display
    show(u)

    # Green strain tensor
    # (https://www.continuummechanics.org/greenstrain.html)
    E = U + transpose(U)  # linear term
    #E += np.einsum("ijkl,ijkm->ijlm", U, U)  # quadratic term
    E /= 2.0  # engineering convention

    # Cauchy stress tensor
    # (https://en.wikipedia.org/wiki/Hooke%27s_law#Linear_elasticity_theory_for_continuous_media)
    S = (s/(1.0+k)) * E
    S[:, :, [[0, 1]], [[0, 1]]] += (s*k/((1.0+k)*(1.0-2.0*k))) * trace(E)

    # Acceleration from conservation of momentum
    a = g*f + (divergence(S) - c*v)/m

    # Temporal integration
    v += a*dt
    u += v*dt + 0.5*a*dt**2
    t += dt

    # Update score
    score += judge()

    # Real time throttle
    remaining_time = (1000//rate) - (pygame.time.get_ticks() - start_time)
    if remaining_time > 0:
        pygame.time.wait(remaining_time)

print("Finished! Your final score was: {}".format(score))
