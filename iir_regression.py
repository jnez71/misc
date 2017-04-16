"""
Tunes a linear IIR filter to have any specified impulse response.
The desired impulse response is specified over some finite time window.
Then the matched IIR filter can be said to "predict" the rest of this
function at times outside the window by simple simulation.
Since this tuning is not for a real-time application, instead of using
an iterative "adaptive filter" method, analytical least-squares was used.

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# Function for forming an IIR-filter regression matrix given data x and order M
regmat = lambda x, M: np.hstack((np.ones((len(x), 1)), toeplitz(x, np.concatenate(([x[0]], np.zeros(M-1))))))

# Function that takes data x and regression matrix X and returns the autocor, X.T.dot(X), and crosscor, X.T.dot(x) for x = x[1:].
cormats = lambda x, X: (X.T.dot(X), X.T.dot(x[1:]))

# Weird function to predict, take your pick
func = lambda t: np.arctan(np.cos(-0.01*t)+np.cos(0.01*t+2)+(np.sin(0.01*1/(t+10000)-np.pi/2)+np.sin(0.01*t)*np.sin(0.1*t)**2*np.exp(np.cos(0.001*np.sign(t-100)*np.sin(t)))))
# func = lambda t: 0.1*np.random.randn(np.shape(t)[0]) + np.arctan(np.cos(-0.01*t)+np.cos(0.01*t+2)+(np.sin(0.01*1/(t+10000)-np.pi/2)+np.sin(0.01*t)*np.sin(0.1*t)**2*np.exp(np.cos(0.001*np.sign(t-100)*np.sin(t)))))
# func = lambda t: np.cos(1*np.exp(-t/500)*t)

# Training data
samples = 1600
x = func(np.arange(0, samples))

# Filter order
M = 800

# Regression matrix
X = regmat(x[:-1], M)

# Optimal weights
w = npl.pinv(X).dot(x[1:])

####

# Full simulation time
T = 3000
time = np.arange(0, T)

# Prepare to record
y_pred = []
y_pred.extend(x)

# Let the "filter" run "autonomously" for samples beyond the ones it trained on
for i, t in enumerate(np.arange(samples-1, T-1)):
    filtout = w[0]
    for d in np.arange(0, M):
        filtout += w[d+1]*y_pred[t-d]
    y_pred.append(filtout)

print("MSE: {}".format(np.sqrt(np.mean(np.square(np.subtract(func(time[samples:]), y_pred[samples:]))))))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(time, np.concatenate((x, func(time[samples:]))), 'g', linewidth=2, label="Desired Response")
ax.plot(time, y_pred, 'k', linewidth=1, label="Filter's Response")
ax.set_xlabel("Training Time (left) and Prediction Time (right)", fontsize=22)
plt.axvline(samples-1)
plt.legend()
plt.show()
