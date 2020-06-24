#!/usr/bin/env python3
"""
Time-independent Schrodinger equation (TISE) solved by direct
discretization of the eigenvalue problem.

f : x -> C
Hf = Ef
H = (p^2)/(2m) + V
p = -ih(d/dx)

Discretize x. Then f is a finite-dimensional vector.
Construct a finite-dimensional Hermitian matrix representation of H.
Use a numerical eigenvector solver on that matrix.

"""
# Dependencies
import numpy as np
from matplotlib import pyplot

# Display setup
np.set_printoptions(suppress=True)
pyplot.rcParams["font.size"] = 20
pyplot.rcParams["figure.facecolor"] = pyplot.rcParams["axes.facecolor"] = [0.05]*3
pyplot.rcParams["text.color"] = pyplot.rcParams["axes.labelcolor"] = pyplot.rcParams["xtick.color"] = pyplot.rcParams["ytick.color"] = [1.0]*3

##################################################

# Particle
h = 1.0
m = 1.0
a = -h**2 / (2.0*m)

# Domain
L = 1.0
dx = 0.001
x = np.arange(0.0, L, dx)
n = len(x)

# Potential
V = np.zeros(n, float)
V[0] = V[-1] = 1000.0
V[1:n//5] = 500.0

##################################################

# Hamiltonian
def H(f):
    D2f = np.convolve(f, [1.0, -2.0, 1.0], "same") / dx**2  # zero-pad boundary condition implies infinite square-well
    Vf = V*f
    return a*D2f + Vf

# Matrix representation of Hamiltonian
print("Constructing...")
M = np.column_stack([H(f) for f in np.identity(n)])
assert np.allclose(M, M.conj().T)

# Solve eigen-problem
print("Solving...")
E, f = np.linalg.eigh(M)

##################################################

# Visualize
print("Plotting...")
select = 5
scale = np.max(V)
colors = pyplot.cm.cool(np.linspace(0.0, 1.0, select))
figure, axes = pyplot.subplots(2, 1)
figure.suptitle("TISE Solutions")
axes[0].set_ylabel("Stationary States")
for i in range(select):
    axes[0].plot(x, scale*f[:, i], color=colors[i], label="E{0}={1}".format(i, np.round(E[i], 1)))
figure.legend(fontsize=16, loc="upper right")
axes[1].set_ylabel("Potential")
axes[1].plot(x, V, color='r', linewidth=5)
axes[1].grid(True)
axes[-1].set_xlabel("Position")
print("Close figure to finish...")
pyplot.show()
