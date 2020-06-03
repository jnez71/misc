#!/usr/bin/env python3
"""
Understanding SGD vs GD.

"""
from autograd import numpy as np, grad
from matplotlib import pyplot

np.set_printoptions(suppress=True)  # never show scientific-notation
pyplot.rcParams["axes.grid"] = True  # grid on plots by default
pyplot.rcParams["font.size"] = 16  # default fontsize for plot labels

##################################################

# Noisy scalar-polynomial data
X = np.arange(-3, 3, 0.1).reshape(-1, 1)  # n-by-1 matrix (n points, 1 feature)
Y = 1*X**3 + 2*X**2 + 3*X + 4 + np.random.normal(0.0, 2.0, X.shape)  # n-by-1 matrix
n = len(X)

# Scalar-polynomial model with parameters c
def f(c, X):
    return c[0]*X**3 + c[1]*X**2 + c[2]*X + c[3]

##################################################

# L2 loss
def l(c, X, Y):
    return np.mean(np.sum((Y - f(c, X))**2, axis=1))
dldc = grad(l)  # this gradient would not be hard to write by hand, but autodiff is even easier!

# Closed-form optimum for this linear regression (assuming L2 loss)
print("(solving pseudoinverse)")
A = np.concatenate((X**3, X**2, X, np.ones_like(X)), axis=1)  # A(X)*c = Y
c_opt = np.linalg.pinv(A).dot(Y)  # c = (A(X)^~1)*Y  where pseudoinverse projection minimizes L2 error

##################################################

# Gradient descent assuming model f and loss l
def gd(X, Y, c0, step, iters):
    print("(running gd)")
    c = np.array(c0, float)
    L = []
    for i in range(iters):
        c -= step*dldc(c, X, Y)
        L.append(l(c, X, Y))  # record loss at each iter for plotting
    return c, L

# Stochastic gradient descent assuming model f and loss l
def sgd(X, Y, c0, step, iters, batch_size, replace=False):
    print("(running sgd)")
    c = np.array(c0, float)
    L = []
    for i in range(iters * n//batch_size):
        # Randomly select batch_size integers from 0 to n-1 with or without replacement
        # If you want to guarantee that every data-point gets used the same amount,
        # then do what is done in practice: iterate over batches of a shuffle.
        samples = np.random.choice(n, batch_size, replace=replace)
        c -= step*dldc(c, X[samples], Y[samples])
        L.append(l(c, X, Y))  # record TOTAL loss at each epoch for plotting
    return c, L

# On convex problems (like linear regression), SGD is a
# BAD idea because there are no local optima to "shake" out of.
# SGD is useful for nonconvex optimizations where the effective
# added noise might get us out of local optima, and for huge
# datasets where batch-processing (processing in small chunks)
# may be the only feasible option (you can't just load an entire 5TB
# SQL database into Python's working memory, so query batches).

##################################################

# Select hyperparameters
c0 = [0, 0, 0, 0]  # selecting a good random initial condition is not as important for convex problems
step = 0.008
iters = 200
batch_size = 15

# Run iterative optimizers
c_gd, L_gd = gd(X, Y, c0, step, iters)  # same as SGD with batch_size=n (and replace=False)
c_sgd, L_sgd = sgd(X, Y, c0, step, iters, batch_size)

# Compare results
print("c_opt:", c_opt.flatten())
print(" c_gd:", c_gd)
print("c_sgd:", c_sgd)
fig, axes = pyplot.subplots(1, 2)
axes[0].set_title("Fits")
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X")
axes[0].scatter(X, Y, c='k', alpha=0.4, s=16, label="Data")
axes[0].plot(X, f(c_opt, X), c='g', ls='--', lw=4, label="Opt")
axes[0].plot(X, f(c_gd, X), c='b', label="GD")
axes[0].plot(X, f(c_sgd, X), c='r', label="SGD")
axes[0].legend()
axes[1].set_title("Convergence")
axes[1].set_ylabel("Loss")
axes[1].set_xlabel("Iteration")
axes[1].plot(L_gd, c='b', alpha=0.6, lw=3, label="GD")
axes[1].plot(L_sgd, c='r', label="SGD")
axes[1].set_ylim([0, l(c0, X, Y)])
axes[1].set_xlim([0, iters])
axes[1].legend()
pyplot.show()
