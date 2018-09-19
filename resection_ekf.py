#!/usr/bin/env python2
"""
Solving a noisy 2D-position-resection problem with an extended Kalman filter.
https://en.wikipedia.org/wiki/Position_resection
https://en.wikipedia.org/wiki/Kalman_filter

"""
from __future__ import division
import numpy as np; npl = np.linalg
from matplotlib import pyplot
from matplotlib.patches import Ellipse
from matplotlib import animation

##################################################

# Unknown true state of the navigator [position, heading]
state_true = np.array([4, 2, np.deg2rad(31)])

# Known landmark positions (assume exact)
landmarks = np.array([[ 5,  8],
                      [-4,  1],
                      [ 4,  0],
                      [ 7,  4],
                      [ 2,  5],
                      [-4, -7],
                      [ 0, 10]])

##################################################

# Bearing sensor noise covariance
sensor_covar = (np.deg2rad(5)**2) * np.eye(len(landmarks))

# State prior mean and covariance
prior_mean = np.array([-1, 1, 0])
prior_covar = np.diag([5, 5, np.deg2rad(150)])**2

##################################################

# Unwraps angles onto [-pi, pi]
def unwrap(angle):
    return np.mod(angle+np.pi, 2*np.pi) - np.pi

# Ideal bearing sensor model
def sensor_model(state):
    deltas = landmarks - state[:2]
    return unwrap(np.arctan2(deltas[:,1], deltas[:,0]) - state[2])

# Bearing sensor model derivative with respect to state
def sensor_jac(state):
    deltas = landmarks - state[:2]
    ddirs = deltas / np.sum(deltas**2, axis=1).reshape(-1, 1)
    return np.stack((ddirs[:,1], -ddirs[:,0], -1*np.ones(len(landmarks))), axis=1)

##################################################

# Acquire some measurements
noise = np.random.multivariate_normal(np.zeros(len(landmarks)), sensor_covar)
measurements = sensor_model(state_true) + noise

# Using a linearized model of our measurements, update our prior
# via the best (minimum covariance) linear estimator of the estimate error
posterior_mean = np.copy(prior_mean)
posterior_covar = np.copy(prior_covar)
def update():
    global prior_mean, posterior_mean, posterior_covar
    slope = sensor_jac(prior_mean)
    cross_covar = prior_covar.dot(slope.T)
    measurement_covar = slope.dot(prior_covar).dot(slope.T) + sensor_covar
    projection = cross_covar.dot(npl.inv(measurement_covar))
    posterior_mean = prior_mean + projection.dot(unwrap(measurements - sensor_model(prior_mean)))
    posterior_covar = prior_covar - projection.dot(slope).dot(prior_covar)
    prior_mean = posterior_mean

################################################## DEMONSTRATION

fig = pyplot.figure()
ax = fig.add_subplot(1, 1, 1)
ax.grid(True)
ax.axis("equal")

setup_calls = 0
def plot_setup():
    global ellipse, estim_point, estim_head, conf, setup_calls
    setup_calls += 1
    if setup_calls == 1:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='g', label="landmark")
        ax.scatter(state_true[0], state_true[1], c='b', label="truth")
        ax.quiver(state_true[0], state_true[1], np.cos(state_true[2]), np.sin(state_true[2]), color='b', scale=20, width=5e-3)
        for landmark, measurement in zip(landmarks, measurements):
            dist = npl.norm(landmark - state_true[:2])
            ang = unwrap(measurement + state_true[2])
            hit = state_true[:2] + [dist*np.cos(ang), dist*np.sin(ang)]
            ax.plot([state_true[0], hit[0]], [state_true[1], hit[1]], color='y', linewidth=1)
        ax.plot([0, 0], [0, 0], color='y', linewidth=1, label="measurement")
        estim_point = ax.scatter(prior_mean[0], prior_mean[1], c='r', label="estimate")
        estim_head = ax.quiver(prior_mean[0], prior_mean[1], np.cos(prior_mean[2]), np.sin(prior_mean[2]), color='r', scale=20, width=5e-3)
        conf = 2*np.sqrt(6)
        evals, evecs = npl.eigh(prior_covar[:2, :2])
        ellipse = Ellipse(xy=prior_mean, width=conf*np.sqrt(evals[-1]), height=conf*np.sqrt(evals[0]),
                          angle=np.rad2deg(np.arctan2(evecs[1, -1], evecs[0, -1])),
                          edgecolor='r', lw=1, facecolor='none')
        ax.add_artist(ellipse)
        ax.plot([0, 0], [0, 0], color='r', linewidth=2, label="95% confidence")
        ax.legend(loc=2)
    return []

def callback(iter):
    global ellipse, estim_point, estim_head
    ellipse.remove()
    estim_point.remove()
    estim_head.remove()
    evals, evecs = npl.eigh(posterior_covar[:2, :2])
    ellipse = Ellipse(xy=prior_mean, width=conf*np.sqrt(evals[-1]), height=conf*np.sqrt(evals[0]),
                      angle=np.rad2deg(np.arctan2(evecs[1, -1], evecs[0, -1])),
                      edgecolor='r', lw=1, facecolor='none')
    ax.add_artist(ellipse)
    estim_point = ax.scatter(prior_mean[0], prior_mean[1], c='r', label="estimate")
    estim_head = ax.quiver(prior_mean[0], prior_mean[1], np.cos(prior_mean[2]), np.sin(prior_mean[2]), color='r', scale=20, width=5e-3)
    update()
    return []

fps = 1
ani = animation.FuncAnimation(fig, callback, range(10), repeat=False, init_func=plot_setup, interval=1000/fps, blit=True)
record = False
if record:
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
    ani.save("resection_ekf.mp4", writer=writer)
else:
    print("Close figure window to finish...")
    pyplot.show()
