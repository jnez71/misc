#!/usr/bin/env python3
"""
Time-independent Schrodinger equation (TISE) solved by direct
discretization of the eigenvalue problem.

f : x -> C
Hf = Ef
H = (p^2)/(2m) + V
p = -jh(d/dx)

Discretize x. Then f is a finite-dimensional vector.
Construct a finite-dimensional Hermitian matrix representation of H.
Use a numerical eigen-solver on that matrix.

"""
# Dependencies
import numpy as np
from matplotlib import pyplot

################################################## COMPUTATION

# Domain
L = 2.0  # spatial extent
dx = 0.002  # spatial stepsize
x = np.arange(-L/2.0, L/2.0, dx)  # position values
n = len(x)  # dimensionality

# Properties
h = 1.0  # reduced Planck's constant
m = 1.0  # particle mass
a = -h**2 / (2.0*m)  # combined coefficient

# Potential (later modifiable by user through GUI)
V = 0.5*m*(20.0*x)**2  # harmonic oscillator

# Hamiltonian (depends on globally defined domain, properties, and potential)
def H(f):
    D2f = np.convolve(f, [1.0, -2.0, 1.0], "same") / dx**2  # Laplacian (note zero-pad boundary condition implies infinite square-well)
    Vf = V*f  # pointwise multiplication of potential
    return a*D2f + Vf  # Hf

# Solver (depends on globally defined Hamiltonian)
def solve_TISE():
    M = np.column_stack([H(f) for f in np.identity(n)])  # build matrix representation of Hamiltonian by testing on standard basis
    E, f = np.linalg.eigh(M)  # let a modern eigen-solver do the hard work on our fully discretized problem
    return E, f
E, f = solve_TISE()

################################################## VISUALIZATION

# Visualization parameters
levels = [0, 1, 2, 3, 4]  # which energy levels to display
colors = pyplot.cm.cool(np.linspace(0.0, 1.0, len(levels)))  # how to color-code the energy levels
impact = 3  # percent of domain to be affected by a single click
Vmax = 200.0  # maximum specifiable potential value

# Global visualization setup
np.set_printoptions(suppress=True)  # don't use scientific notation
pyplot.rcParams["font.size"] = 20
pyplot.rcParams["figure.facecolor"] = pyplot.rcParams["axes.facecolor"] = [0.05]*3  # dark background
pyplot.rcParams["text.color"] = pyplot.rcParams["axes.labelcolor"] = pyplot.rcParams["xtick.color"] = pyplot.rcParams["ytick.color"] = [1.0]*3  # light text

# Figure setup
figure, axes = pyplot.subplots(2, 1)
figure.suptitle("TISE Solutions")
axes[0].set_ylabel("Stationary States")
axes[1].set_ylabel("Potential")
axes[-1].set_xlabel("Position")
axes[1].grid(True)

# Plotter (relies on globally defined solution and figure)
def plot_TISE():
    global graphics  # remember graphics objects for later modifying
    graphics = [axes[0].plot(x, f[:, i], color=colors[i], label="E{0}={1}".format(i, np.round(E[i], 1)))[0] for i in levels]  # plot solutions
    graphics.append(figure.legend(fontsize=16, loc="upper right"))  # legend shows energy values
    graphics.append(axes[1].plot(x, V, color='r', linewidth=5)[0])  # plot potential
    graphics.append(axes[1].axvline(x[0], color='r', linewidth=5))  # plot vertical lines at boundaries...
    graphics.append(axes[1].axvline(x[-1], color='r', linewidth=5))  # ... to remind user of intrinsic infinite square-well
    axes[1].set_ylim([0.0, Vmax])
    figure.canvas.draw()
plot_TISE()

# Callback for user input events (relies on globally defined solver and plotter)
def gui(event):
    global V, E, f, graphics  # user modifies V, then E and f are re-solved, then graphics are replotted
    if (event.inaxes is not axes[1]) or (event.xdata<x[0] or event.xdata>x[-1]):
        return  # user must click in valid plot and region
    if(event.button._name_ == "RIGHT"):
        V = np.zeros(n, float)  # clear on right-click
    elif(event.button._name_ == "LEFT"):
        c = int((event.xdata - x[0] + dx/2.0) / dx)  # domain index of user click
        lo, hi = np.clip([c-n*impact//100, c+n*impact//100], 0, n-1)  # surrounding indexes of V to modify
        V[lo:hi] = np.clip(event.ydata, 0.0, Vmax)  # modify potential
    E, f = solve_TISE()  # re-solve
    for graphic in graphics: graphic.remove()  # remove old graphics
    plot_TISE()  # plot new solution

# Show display with GUI active
print("\nLeft-click to modify the potential field!")
print("Right-click to clear the potential field.")
print("Your cursor must move between clicks to trigger the redraw.")
print("Close the figure to quit.\n")
figure.canvas.mpl_connect("button_press_event", gui)
pyplot.show()

##################################################
