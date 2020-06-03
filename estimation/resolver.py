#!/usr/bin/env python2
"""
Using an EKF for dynamic model estimation of a motor
sensed with only a cheap encoder. Use the model for
both state estimation predict and controller feedforward.

"""
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt

# Motor dynamics
# (model-augmented state is [pos, vel, drag/inertia, b/inertia])
n_x = 4
dyn_noise_mean = np.array([0, 0, 0, 0])
dyn_noise_cov = np.array([[0,    0,    0,    0],
                          [0, 1E-4,    0,    0],
                          [0,    0, 1E-4,    0],
                          [0,    0,    0, 1E-6]])
def dynamic(x, u, dt, add_noise):
    xdot = np.array([x[1],
                     x[3]*u - x[2]*x[1],
                     0,
                     0])
    xnext = x + xdot*dt
    if xnext[2] < 0.1: xnext[2] = 0.1
    if xnext[3] < 0.1: xnext[3] = 0.1
    if add_noise:
        xnext += np.random.multivariate_normal(dyn_noise_mean, dyn_noise_cov)
    return xnext

# Encoder (absolute)
res = 256/360  # ticks/deg
y_per_t = 50  # samples/s
def sense(x):
    return np.floor(res*x[0])

# Jacobians
def dynamic_jac(x, u, dt):
    return np.array([[0,     1,     0, 0],
                     [0, -x[2], -x[1], u],
                     [0,     0,     0, 0],
                     [0,     0,     0, 0]])*dt + np.eye(4)
def sense_jac(x):
    return np.array([[res, 0, 0, 0]])  # like a subgradient of the discontinuous encoder function

# Estimated process and sensor covariances
Q = np.diag([0, 1E-4, 1E-4, 1E-6])
R = np.array([1.0])  # encoder noise should really be uniform between ticks but Kalman says whatevs brah

# Time domain
T = 60  # s
dt = 0.01  # s
t = np.arange(0, T, dt)  # s
i_per_y = int(1/(y_per_t*dt))  # iters/sample

# State, estimate, covariance, and measurement timeseries
x = np.zeros((len(t), n_x))
xh = np.zeros((len(t), n_x))
P = np.zeros((len(t), n_x, n_x))
y = np.zeros((len(t), 1))

# Initial conditions
x[0] = [5, 0, 5, 2]
xh[0] = [0, 0, 1, 1]
P[0] = 50*np.eye(n_x)

# Desired trajectory
# z = (np.ones((len(t), 2)) * [180, 0])
zw = 0.5
z = 15*np.vstack((np.sin(zw*t), zw*np.cos(zw*t))).T

# Feedback gains
Kc = np.array([1, 1])

# Simulation
for i, ti in enumerate(t[1:]):

    # Compute control
    uz = (1/xh[i, 3]) * ((z[i+1, 1] - z[i, 1])/dt + xh[i, 2]*z[i, 1])  # feedforward
    u = Kc.dot(z[i] - x[i, :2]) + uz  # feedback + feedforward

    # Step forward state
    x[i+1] = dynamic(x[i], u, dt, add_noise=True)

    # Step forward estimate ("predict")
    xh[i+1] = dynamic(xh[i], u, dt, add_noise=False)

    # Instantaneous dynamics jacobian
    F = dynamic_jac(xh[i], u, dt)

    # Step forward covariance
    P[i+1] = F.dot(P[i]).dot(F.T) + Q

    # When new measurement comes in...
    if i % i_per_y == 0:

        # Get new measurement
        y[i+1] = sense(x[i+1])

        # Instantaneous sensor jacobian
        H = sense_jac(xh[i+1])

        # Kalman gain
        S = H.dot(P[i+1]).dot(H.T) + R
        K = P[i+1].dot(H.T).dot(1/S)

        # Update state estimate and covariance ("correct")
        xh[i+1] = xh[i+1] + K.flatten()*(y[i+1] - sense(xh[i+1]))
        P[i+1] = (np.eye(n_x) - K.dot(H)).dot(P[i+1])

    # ...otherwise hold last measurement (for plotting only)
    else:
        y[i+1] = np.copy(y[i])

#### Plertz

fig1 = plt.figure()
fig1.suptitle("Motor Estimation and Control via Online EKF-Fitted Model", fontsize=22)
ax1 = fig1.add_subplot(5, 1, 1)
ax1.plot(t, x[:, 0], label="true", color='g', lw=1)
ax1.plot(t, xh[:, 0], label="estimate", color='k', ls=':', lw=3)
ax1.plot(t, z[:, 0], label="desired", color='r', ls='--', lw=1)
ax1.set_ylabel("position", fontsize=18)
ax1.legend(loc='upper right')
ax1.grid(True)

ax2 = fig1.add_subplot(5, 1, 2, sharex=ax1)
ax2.plot(t, x[:, 1], label="true", color='g', lw=1)
ax2.plot(t, xh[:, 1], label="estimate", color='k', ls=':', lw=3)
ax2.plot(t, z[:, 1], label="desired", color='r', ls='--', lw=1)
ax2.set_ylabel("velocity", fontsize=18)
ax2.grid(True)

ax3 = fig1.add_subplot(5, 1, 3, sharex=ax1)
ax3.plot(t, x[:, 2], label="true", color='g', lw=1)
ax3.plot(t, xh[:, 2], label="estimate", color='k', ls=':', lw=3)
ax3.set_ylabel("drag/inertia", fontsize=18)
ax3.grid(True)

ax4 = fig1.add_subplot(5, 1, 4, sharex=ax1)
ax4.plot(t, x[:, 3], label="true", color='g', lw=1)
ax4.plot(t, xh[:, 3], label="estimate", color='k', ls=':', lw=3)
ax4.set_ylabel("b/inertia", fontsize=18)
ax4.grid(True)

ax5 = fig1.add_subplot(5, 1, 5, sharex=ax1)
ax5.plot(t, y, label="measured", color='b', lw=1)
ax5.set_ylabel("encoder\ndata", fontsize=18)
ax5.set_xlabel("time (s)", fontsize=18)
ax5.set_xlim([0, T])
ax5.grid(True)

plt.show()
