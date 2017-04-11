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
nS = 3
nA = 2
nM = 2
nT = int(2E4)

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

# Q-learning basis-form and learning-rate structure
basis = "linear"  # "linear", "quadratic", or "radial"
gain_type = "kalman"  # "static", "decay", "kalman"

# Q approximation function: linear + offset
if basis == "linear":
    nF = 5
    def F(b, u):
        f = [b[0], b[1]]
        if u == 1: return np.concatenate((f, np.zeros_like(f), [1]))
        return np.concatenate((np.zeros_like(f), f, [0]))

# Q approximation function: quadratic (but no cross terms) + offset
elif basis == "quadratic":
    nF = 9
    def F(b, u):
        f = [b[0], b[0]**2, b[1], b[1]**2]
        if u == 1: return np.concatenate((f, np.zeros_like(f), [1]))
        return np.concatenate((np.zeros_like(f), f, [0]))

# Q approximation function: radial (with centers z in Z) + offset
elif basis == "radial":
    Z = np.array([[0.8, 0.1],
                  [0.1, 0.1]])
    nF = 2*len(Z)+1
    def F(b, u):
        f = np.exp(-npl.norm(Z - [b[0], b[1]], axis=1)/2)
        f = f / np.sum(f)
        if u == 1: return np.concatenate((f, np.zeros_like(f), [1]))
        return np.concatenate((np.zeros_like(f), f, [0]))

# State, measurement, belief, cost, and parametric-Q histories
x = np.zeros(nT, dtype=np.int64)
y = np.zeros(nT, dtype=np.int64)
b = np.zeros((nT, nS), dtype=np.float64)
c = np.zeros(nT, dtype=np.float64)
f = np.zeros((nT, nF), dtype=np.float64)
q = np.zeros((nT, nF), dtype=np.float64)

# Initial conditions
x[0] = 0
b[0] = [1/3, 1/3, 1/3]
f[0] = F(b[0], 0)
q[0] = 200*(np.random.rand(nF)-0.5)
K = np.eye(nF)

# Function for randomly sampling with a given discrete probability density
sample_from = lambda p: np.argwhere(np.random.sample() < np.cumsum(p))[0][0]

# Simulation
T = np.arange(nT)
for t in T[1:]:

    # Randomly choose action, accept true cost
    u = sample_from([0.5, 0.5])
    c[t] = C[u, x[t-1]]

    # Advance state, obtain measurement
    x[t] = sample_from(P[u, x[t-1]])
    y[t] = sample_from(R[u, x[t]])

    # Update belief
    b[t] = (b[t-1].dot(P[u]))*R[u, :, y[t]]
    b[t] = b[t] / np.sum(b[t])

    # Approximate Bellman error and jacobian
    f_a = np.array([F(b[t], a) for a in xrange(nA)])
    E = b[t-1].dot(C[u]) + g*np.min(f_a.dot(q[t-1])) - f[t-1].dot(q[t-1])
    K = ((t-1)*K + np.outer(f[t-1], f[t-1]))/t

    # Update Q approximation
    if gain_type == "static":
        q[t] = q[t-1] + 0.001*E*f[t-1]
    elif gain_type == "decay":
        q[t] = q[t-1] + (100/t)*E*f[t-1]
    elif gain_type == "kalman":
        q[t] = q[t-1] + 0.001*npl.pinv(K).dot(E*f[t-1])
    f[t] = f_a[u]

    # Heartbeat
    if t % int(nT/10) == 0:
        print("Progress: {}%".format(int(100*t/nT)))
        print("Bellman Error: {}".format(np.round(E, 3)))
        print("Eigs: {}\n".format(np.round(npl.eigvals(K), 3)))
print("Final q: {}".format(np.round(q[-1], 3)))
np.save("q", q[-1])

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
        Q[a, i] = F(bi, a).dot(q[-1])
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
try: ax.scatter(Z[:, 0], Z[:, 1], 1-np.sum(Z, axis=1), c='g', s=32, zorder=10)
except: pass
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
