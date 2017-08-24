"""
Demonstration of the benefit of local-linearization integration for solving
ODE's. Say goodbye to Euler and RK4.

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

dynamics = lambda x: np.array([-x[1]*x[0],
                               -x[0]])

solution = lambda t: np.array([2*t**-2,
                               2*t**-1])

# Basic second time-derivative information
dynamicsdot = lambda x: np.array([x[0]**2 + x[0]*x[1]**2,
                                  x[1]*x[0]])

# ...or approximate dynamics(x) around x0 by dynamics_jacobian(x0).dot(x) + dynamics_constant(x0)
dynamics_jacobian = lambda x: np.array([[-x[1], -x[0]],
                                        [   -1,    0]])
dynamics_constant = lambda x: dynamics(x) - dynamics_jacobian(x).dot(x)

dt = 0.1
time = np.arange(0.8, 10, dt)
x0 = solution(time[0])

x_exact = np.zeros((len(time), len(x0)))
x_euler = np.zeros((len(time), len(x0)))
x_ll = np.zeros((len(time), len(x0)))
x_euler2 = np.zeros((len(time), len(x0)))

x_exact[0] = x0
x_euler[0] = x0
x_euler2[0] = x0
x_ll[0] = x0

for i, t in enumerate(time[1:]):
    x_exact[i+1] = solution(t)
    x_euler[i+1] = x_euler[i] + dynamics(x_euler[i])*dt
    x_euler2[i+1] = x_euler2[i] + dynamics(x_euler2[i])*dt + (1/2)*dynamicsdot(x_euler2[i])*dt**2
    x_ll[i+1] = spl.expm(np.vstack([
                    np.hstack([dynamics_jacobian(x_ll[i]), dynamics_constant(x_ll[i]).reshape((2, 1))]),
                    [[0, 0, 0]]]) * dt).dot(np.hstack([x_ll[i], [1]]))[:2]

error_euler = npl.norm(x_exact - x_euler, axis=1)
error_euler2 = npl.norm(x_exact - x_euler2, axis=1)
error_ll = npl.norm(x_exact - x_ll, axis=1)

plt.figure()
plt.grid(True)
plt.plot(x_exact[:, 0], x_exact[:, 1], c='k', label='exact')
plt.plot(x_euler[:, 0], x_euler[:, 1], c='b', label='euler')
plt.plot(x_euler2[:, 0], x_euler2[:, 1], c='g', label='euler2')
plt.plot(x_ll[:, 0], x_ll[:, 1], c='r', label='LL')
plt.legend(loc="lower right", fontsize=18)
plt.title("Solutions to x1dot = -x2*x1 and x2dot = -x1", fontsize=26)
plt.xlabel("x0", fontsize=26)
plt.ylabel("x1", fontsize=26)

plt.figure()
plt.grid(True)
plt.plot(time, error_euler, c='b', label='euler error')
plt.plot(time, error_euler2, c='g', label='euler2 error')
plt.plot(time, error_ll, c='r', label='LL error')
plt.legend(loc="upper left", fontsize=18)
plt.title("Error Norms", fontsize=26)
plt.xlabel("time", fontsize=26)
plt.ylabel("||error||", fontsize=26)

plt.show()
