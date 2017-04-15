"""
Q-learning for a POMDP. Not the best approach, but fun.

"""
# Computational dependencies
from __future__ import division
import numpy as np; npl = np.linalg

# Visualization dependencies
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm as colormap
from mpl_toolkits.mplot3d import Axes3D

# State, action, measurement, and time cardinalities
nS = 3; nA = 2; nM = 2; nT = int(1E5)

# Process Model: transition conditional-probability matrix, nA by nS by nS'
P = np.array([[[  1,   0,   0],
               [  1,   0,   0],
               [  0, 0.3, 0.7]],
              [[0.4,   0, 0.6],
               [0.1, 0.6, 0.3],
               [  0, 0.1, 0.9]]], dtype=np.float64)

# Sensor Model: observation conditional-probability matrix, nA by nS' by nM'
R = np.array([[[    1,   0],
               [    1,   0],
               [1-0.1, 0.1]],
              [[    1,   0],
               [    1,   0],
               [    0,   1]]], dtype=np.float64)

# Discount factor and cost function matrix for exact action-state
# pair, nA by nS (belief-state cost function is <b(*),C(u,*)>)
g = 0.8
C = np.array([[-1, -1, -3],
              [ 0,  0, -2]], dtype=np.float64)

# Q-approximation basis choice...
basis = "planar"

# ...planar
if basis == "planar":
    nF = 8
    def F(b, y, u):
        f = np.zeros(nF)
        if y == 0:
            f[0] = b[0]
            f[1] = b[0]*u
            f[2] = b[2]
            f[3] = b[2]*u
            f[4] = u
        else:
            f[5] = b[2]
            f[6] = u
        f[7] = 1
        return f

# ...or radial
elif basis == "radial":
    Z = np.array([[0.8, 0.1],
                  [0.1, 0.1],
                  [0.1, 0.8]])
    nF = 2*len(Z)+3
    def F(b, y, u):
        f = np.zeros(nF)
        if y == 0:
            f[:len(Z)] = np.exp(-npl.norm(Z - [b[0], b[2]], axis=1)/2)
            f[len(Z):2*len(Z)] = u*f[:len(Z)]
        else:
            f[-3:-1] = (b[2], u)
        f[-1] = 1
        return f

# State, measurement, belief, and parametric-Q histories
x = np.zeros(nT, dtype=np.int64)
y = np.zeros(nT, dtype=np.int64)
b = np.zeros((nT, nS), dtype=np.float64)
f = np.zeros((nT, nF), dtype=np.float64)
q = np.zeros((nT, nF), dtype=np.float64)

# Initial conditions
u = 0
x[0] = 0
b[0] = [1/3, 1/3, 1/3]
f[0] = F(b[0], 0, u)
q[0] = 20*(np.random.rand(nF)-0.5)
Ksum = np.zeros((nF, nF))

# Function for randomly sampling with a given discrete probability density
sample_from = lambda p: np.argwhere(np.random.sample() < np.cumsum(p))[0][0]

# Simulation
T = np.arange(nT)
for t in T[1:]:

    # Randomly choose next action
    ut = sample_from([0.5, 0.5])

    # Advance state, obtain measurement
    x[t] = sample_from(P[u, x[t-1]])
    y[t] = sample_from(R[u, x[t]])

    # Update belief
    b[t] = (b[t-1].dot(P[u]))*R[u, :, y[t]]
    b[t] = b[t] / np.sum(b[t])

    # Approximate error and jacobian
    f_a = np.array([F(b[t], y[t], a) for a in xrange(nA)])
    E = b[t-1].dot(C[u]) + g*np.min(f_a.dot(q[t-1])) - f[t-1].dot(q[t-1])
    Ksum = Ksum + np.outer(f_a[ut], f_a[ut])

    # Update Q approximation
    condition = npl.cond(Ksum)
    if condition < 10000:
        q[t] = q[t-1] + (1/t)*npl.inv(Ksum/t).dot(E*f[t-1])
    else:
        q[t] = q[t-1] + 0.001*E*f[t-1]
    f[t] = f_a[ut]
    u = ut

    # Heartbeat
    if t % int(nT/10) == 0:
        print("Progress: {}%".format(int(100*t/nT)))
        print("Error: {}".format(np.round(E, 3)))
        print("Eigs: {}".format(np.round(npl.eigvals(Ksum/t), 3)))
        print("Condition: {}\n".format(condition))
print("Final q: {}".format(np.round(q[-1], 3)))

# Compute discretized final Q function
res = 81
B = np.vstack((np.repeat(np.linspace(0, 1, res), res),
               np.tile(np.linspace(0, 1, res), res),
               np.zeros(res**2))).T
B = np.delete(B, np.argwhere(np.sum(B, axis=1) > 1).flatten(), axis=0)
B[:, 2] = 1 - np.sum(B, axis=1)
Q = np.zeros((nA, len(B)))
for a in xrange(nA):
    for i, bi in enumerate(B):
        Q[a, i] = F(bi, 0, a).dot(q[-1])
U = np.argmin(Q, axis=0)
V = np.min(Q, axis=0)

# Prepare plots
rcParams.update({'figure.autolayout': True})
fig = plt.figure()
fig.canvas.set_window_title("qrover_results")
nplotsr = 2; nplotsc = 3
fontsize = 16
dens = int(np.ceil(nT/1000))

# Plot policy on belief simplex
ax = fig.add_subplot(nplotsr, nplotsc, 1, projection='3d')
ax.scatter(B[:, 0], B[:, 1], B[:, 2], c=U, zorder=1)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
ax.set_title("Policy", fontsize=fontsize)
ax.set_xlabel("Bottom", fontsize=fontsize)
ax.set_ylabel("Middle", fontsize=fontsize)
ax.set_zlabel("Top", fontsize=fontsize)
ax.view_init(20, 20)

# Plot Q function over belief state
ax = fig.add_subplot(nplotsr, nplotsc, 2, projection='3d')
ax.plot_trisurf(B[::1, 0], B[::1, 1], Q[0, ::1], cmap=colormap.Blues, linewidth=0, antialiased=True)
ax.plot_trisurf(B[::1, 0], B[::1, 1], Q[1, ::1], cmap=colormap.autumn, linewidth=0, antialiased=True)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.set_title("Q Function", fontsize=fontsize)
ax.set_xlabel("Bottom", fontsize=fontsize)
ax.set_ylabel("Middle", fontsize=fontsize)
ax.set_zlabel("Q", fontsize=fontsize)
ax.view_init(20, 20)

# Plot value function over belief state
ax = fig.add_subplot(nplotsr, nplotsc, 3, projection='3d')
ax.plot_trisurf(B[::1, 0], B[::1, 1], V[::1], cmap=colormap.coolwarm, linewidth=0, antialiased=True)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.set_title("Value Function", fontsize=fontsize)
ax.set_xlabel("Bottom", fontsize=fontsize)
ax.set_ylabel("Middle", fontsize=fontsize)
ax.set_zlabel("V", fontsize=fontsize)
ax.view_init(20, 20)

# Plot projection weights
ax = fig.add_subplot(2, 1, 2)
ax.plot(T, q)
ax.set_xlim([0, nT])
ax.set_xlabel("Time", fontsize=fontsize)
ax.set_ylabel("Weights", fontsize=fontsize)
ax.grid(True)

# # Plot belief state exploration
# fig = plt.figure()
# fig.canvas.set_window_title("qrover_results_aux")
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(b[::dens, 0], b[::dens, 1], b[::dens, 2], c='b', alpha=0.1)
# ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
# ax.set_title("Belief-State Exploration", fontsize=fontsize)
# ax.set_xlabel("Bottom", fontsize=fontsize)
# ax.set_ylabel("Middle", fontsize=fontsize)
# ax.set_zlabel("Top", fontsize=fontsize)
# ax.view_init(20, 20)

plt.show()
