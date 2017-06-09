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
                          [0,    0, 1E-3,    0],
                          [0,    0,    0, 1E-6]])
def dynamic(x, u, dt):
    xdot = np.array([x[1],
                     x[3]*u - x[2]*x[1],
                     0,
                     0])
    xnext = x + xdot*dt + np.random.multivariate_normal(dyn_noise_mean, dyn_noise_cov)
    if xnext[2] < 0.1: xnext[2] = 0.1 + 0.001*np.random.sample()
    if xnext[3] < 0.1: xnext[3] = 0.1 + 0.001*np.random.sample()
    return xnext

# Encoder
res = 512/360  # ticks/deg
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
    return np.array([[res, 0, 0, 0]])  # this is a better linear model than dh/dx=0

# Kalman's knobs
Q = np.diag([0, 1E-4, 1E-3, 1E-6])
R = np.array([1])

# Time domain
T = 100  # s
dt = 0.01  # s
t = np.arange(0, T, dt)  # s
i_per_y = int(1/(y_per_t*dt))  # iters/sample

# State, estimate, covariance, and measurement timeseries
x = np.zeros((n_x, len(t)))
xh = np.zeros((n_x, len(t)))
P = np.zeros((n_x, n_x, len(t)))
y = np.zeros((1, len(t)))

# Initial conditions
x[:, 0] = [5, 0, 5, 2]
xh[:, 0] = [0, 0, 1, 1]
P[:, :, 0] = 10*np.eye(n_x)

# Desired trajectory
# z = (np.ones((len(t), 2)) * [180, 0]).T
zw = 0.5
z = 15*np.vstack((np.sin(zw*t), zw*np.cos(zw*t)))

# Feedback gains
Kc = np.array([1, 1])

# Simulation
for i, ti in enumerate(t[1:]):

    # Compute control
    uz = (1/xh[3, i]) * ((z[1, i+1] - z[1, i])/dt + xh[2, i]*z[1, i])
    u = Kc.dot(z[:, i] - x[:2, i]) + uz

    # Step forward state
    x[:, i+1] = dynamic(x[:, i], u, dt)

    # Step forward estimate ("predict")
    xh[:, i+1] = dynamic(xh[:, i], u, dt)

    # Instantaneous dynamics jacobian
    F = dynamic_jac(xh[:, i], u, dt)

    # Step forward covariance
    P[:, :, i+1] = F.dot(P[:, :, i]).dot(F.T) + Q

    # When new measurement comes in...
    if i % i_per_y == 0:

        # Get new measurement
        y[:, i+1] = sense(x[:, i+1])

        # Instantaneous sensor jacobian
        H = sense_jac(xh[:, i+1])

        # Kalman gain
        S = H.dot(P[:, :, i+1]).dot(H.T) + R
        K = P[:, :, i+1].dot(H.T).dot(1/S)

        # Update state estimate and covariance ("correct")
        xh[:, i+1] = xh[:, i+1] + K.flatten()*(y[:, i+1] - sense(xh[:, i+1]))
        P[:, :, i+1] = (np.eye(n_x) - K.dot(H)).dot(P[:, :, i+1])

    # ...otherwise hold last measurement (for plotting only)
    else:
        y[:, i+1] = np.copy(y[:, i])

#### Plertz

fig1 = plt.figure()
fig1.suptitle("Estimation and Tracking via Online EKF-Learned Model", fontsize=22)
ax1 = fig1.add_subplot(4, 1, 1)
ax1.plot(t, x[0, :], label="true", color='g', lw=3)
ax1.plot(t, xh[0, :], label="estimate", color='k', ls=':', lw=3)
ax1.plot(t, z[0, :], label="desired", color='r', ls='--')
ax1.set_ylabel("position\n(deg)", fontsize=18)
ax1.legend(loc='upper right')
ax1.grid(True)

ax1 = fig1.add_subplot(4, 1, 2)
ax1.plot(t, x[1, :], label="true", color='g', lw=3)
ax1.plot(t, xh[1, :], label="estimate", color='k', ls=':', lw=3)
ax1.plot(t, z[1, :], label="desired", color='r', ls='--')
ax1.set_ylabel("velocity\n(deg/s)", fontsize=18)
ax1.grid(True)

ax1 = fig1.add_subplot(4, 1, 3)
ax1.plot(t, x[2, :], label="true", color='g', lw=3)
ax1.plot(t, xh[2, :], label="estimate", color='k', ls=':', lw=3)
ax1.set_ylabel("drag/inertia", fontsize=18)
ax1.grid(True)

ax1 = fig1.add_subplot(4, 1, 4)
ax1.plot(t, x[3, :], label="true", color='g', lw=3)
ax1.plot(t, xh[3, :], label="estimate", color='k', ls=':', lw=3)
ax1.set_ylabel("b/inertia", fontsize=18)
ax1.set_xlabel("time (s)", fontsize=18)
ax1.grid(True)

plt.show()
