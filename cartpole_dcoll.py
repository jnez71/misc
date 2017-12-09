#!/usr/bin/python
"""
Cart-pole toy problem to test direct collocation optimal control.
State: q = [pos, ang, vel, angvel]
Input: u = force

"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Animate realtime (Mayavi) or plot timeseries (Matplotlib)
realtime = True
if realtime:
    from mayavi import mlab
    import time
    import os, vtk
    if os.path.exists("/dev/null"): shadow_realm = "/dev/null"
    else: shadow_realm = "c:\\nul"
    mlab_warning_output = vtk.vtkFileOutputWindow()
    mlab_warning_output.SetFileName(shadow_realm)
    vtk.vtkOutputWindow().SetInstance(mlab_warning_output)
else:
    from matplotlib import pyplot

# Fresh optimization or load from saved?
fresh_optimization = True

############################################################################### SYSTEM

# Dimensions
n_u = 1  # inputs
n_d = 2  # degrees of freedom
n_q = 2*n_d  # states

# Physical parameters
l = 1  # pole length, m
m = [0.3, 0.2]  # cart and pole point masses, kg
b = [0.5, 0.05]  # damping, [N/(m/s), (N*m)/(rad/s)]
g = 9.81  # downwards gravity, m/s^2

# Design-dynamics qdot = f(q, u), ignores limits
def f(q, u):
    return np.array([q[2],
                     q[3],
                     (l*m[1]*np.sin(q[1])*q[3]**2 + u - b[0]*q[2] + m[1]*g*np.cos(q[1])*np.sin(q[1])) / (m[0] + m[1]*(1-np.cos(q[1])**2)),
                     -(l*m[1]*np.cos(q[1])*np.sin(q[1])*q[3]**2 + u*np.cos(q[1]) + (m[0]+m[1])*g*np.sin(q[1]) + b[1]*q[3]) / (l*m[0] + l*m[1]*(1-np.cos(q[1])**2))])

# Vectorized version of design-dynamics
def F(Q, U):
    return np.array([Q[:, 2],
                     Q[:, 3],
                     (l*m[1]*np.sin(Q[:, 1])*Q[:, 3]**2 + U - b[0]*Q[:, 2] + m[1]*g*np.cos(Q[:, 1])*np.sin(Q[:, 1])) / (m[0] + m[1]*(1-np.cos(Q[:, 1])**2)),
                     -(l*m[1]*np.cos(Q[:, 1])*np.sin(Q[:, 1])*Q[:, 3]**2 + U*np.cos(Q[:, 1]) + (m[0]+m[1])*g*np.sin(Q[:, 1]) + b[1]*Q[:, 3]) / (l*m[0] + l*m[1]*(1-np.cos(Q[:, 1])**2))]).T

# Vectorized design-dynamics jacobians
def dFdQ(Q, U):
    pass # ???
def dFdU(Q, U):
    pass # ???

# Initial condition
t = 0  # don't change
q = np.array([-2, 0, 0, 0], dtype=np.float64)

############################################################################### OPTIMIZATION

# Desired final condition
tN = 3  # s
qN = np.array([0, np.pi, 0, 0], dtype=np.float64)

# State and input limits [(min, max)_0, (min, max)_1...]
qlims = [(-3, 3), (None, None), (None, None), (None, None)]
ulims = (-4, 4)

# Performs trapezoidal direct collocation given grid resolution h in time units,
# initial solution guess Z0 = [U0, Q0.flatten()], and other stuff
# Returns (Uopt, Qopt, success_bool)
def dcollopt(h, Z0, q0=np.copy(q), qN=np.copy(qN), tN=tN, qlims=qlims, ulims=ulims, n_q=n_q):

    # Number of grid points
    N = int(tN / h)

    # Objective function and jacobian
    def J(Z):
        U2 = Z[:N]**2
        return (h/2) * (U2[0] + 2*np.sum(U2[1:-1]) + U2[-1])
    def dJdZ(Z):
        return h*np.concatenate(([Z[0]], 2*Z[1:N-1], [Z[N-1]], np.zeros((N*n_q))))

    # Collocation error function and jacobian
    def C(Z):
        U = Z[:N]
        Q = Z[N:].reshape(N, n_q)
        Qdot = F(Q, U)
        return ((h/2)*(Qdot[1:] + Qdot[:-1]) + (Q[:-1] - Q[1:])).flatten()
    def dCdZ(Z):
        pass # ???
    
    # Boundary error functions and jacobians
    def B0(Z):
        return Z[N:N+n_q] - q0
    def dB0dZ(Z):
        return np.hstack((np.zeros((n_q, N)), np.eye(n_q), np.zeros((n_q, (N-1)*n_q))))
    def BN(Z):
        return Z[-n_q:] - qN
    def dBNdZ(Z):
        return np.hstack((np.zeros((n_q, len(Z)-n_q)), np.eye(n_q)))

    # Construct constraints
    coll_constraints = ({"type": "eq", "fun": C},
                        {"type": "eq", "fun": B0, "jac": dB0dZ},
                        {"type": "eq", "fun": BN, "jac": dBNdZ})
    path_constraints = np.vstack(([ulims]*N, qlims*N))

    # Optimize
    opt = minimize(fun=J, x0=Z0, method="SLSQP", jac=dJdZ, bounds=path_constraints,
                   constraints=coll_constraints, tol=1e-3, options={"maxiter": 250, "disp": True})
    return opt.x[:N], opt.x[N:].reshape(N, n_q), opt.success

# Grid refinement sequence
H = [0.4]
while True:
    if H[-1] > 0.03: H.append(H[-1]*0.7)
    elif H[-1] > 0.008: H.append(H[-1]*0.9)
    else: break
print "Grid refinement sequence: {}\n".format(H)

# Do fresh optimization
if fresh_optimization:

    # Initial solution guess
    N0 = int(tN / H[0])
    Uopt = np.zeros(N0)
    Qopt = np.zeros((N0, n_q))
    for i in xrange(n_q):
        Qopt[:, i] = np.linspace(q[i], qN[i], N0)
    Zopt = np.concatenate((Uopt, Qopt.flatten()))
    opt_grid = np.linspace(0, tN, N0)

    # Optimize and refine
    for i in xrange(len(H)):
        print "Grid size: {}".format(H[i])
        Uopt, Qopt, success = dcollopt(H[i], Zopt)
        print ""
        opt_control = interp1d(opt_grid, Uopt, axis=0, kind="linear", bounds_error=False, fill_value=0)
        opt_state = interp1d(opt_grid, Qopt, axis=0, kind="quadratic", bounds_error=False, fill_value=np.copy(qN))
        if i < len(H)-1:  # resample
            opt_grid = np.linspace(0, tN, int(tN / H[i+1]))
            Zopt = np.concatenate((opt_control(opt_grid), opt_state(opt_grid).flatten()))
        np.save("grid.py", opt_grid)
        np.save("Uopt", Uopt)
        np.save("Qopt", Qopt)

# or load previous results
else:
    opt_grid = np.load("grid.npy")
    Uopt = np.load("Uopt.npy")
    Qopt = np.load("Qopt.npy")
    opt_control = interp1d(opt_grid, Uopt, axis=0, kind="linear", bounds_error=False, fill_value=0)
    opt_state = interp1d(opt_grid, Qopt, axis=0, kind="quadratic", bounds_error=False, fill_value=np.copy(qN))

try:
    pyplot.plot(opt_grid, Qopt[:, 0], label="pos")
    pyplot.plot(opt_grid, Qopt[:, 1], label="ang")
    pyplot.plot(opt_grid, Qopt[:, 2], label="vel")
    pyplot.plot(opt_grid, Qopt[:, 3], label="angvel")
    pyplot.plot(opt_grid, Uopt, label="force")
    pyplot.grid(True)
    pyplot.title("Ideal Trajectory", fontsize=16)
    pyplot.xlabel("Time", fontsize=16)
    pyplot.ylabel("Results", fontsize=16)
    pyplot.legend(fontsize=16)
    pyplot.show()
except:
    pass

############################################################################### SIMULATION

# Simulation timestep
dt = 0.005

if realtime:

    # Generate visualization objects
    fig = mlab.figure(size=(500, 500), bgcolor=(0.25, 0.25, 0.25))
    track = mlab.plot3d(qlims[0], (0, 0), (0, 0), line_width=1, color=(1, 1, 1))
    cart = mlab.points3d(q[0], 0, 0, scale_factor=0.2, mode="cube", color=(0, 0, 1))
    pole = mlab.plot3d((q[0], q[0]+l*np.sin(q[1])), (-0.1, -0.1), (0, -l*np.cos(q[1])), line_width=1, color=(0, 1, 0))
    goal = mlab.points3d(qN[0], 0, -1.25*l, scale_factor=0.2, mode="axes", color=(1, 0, 0))
    disp = mlab.text3d(-0.5, 0, 1.5*l, "0.0", scale=0.5)
    mlab.view(azimuth=-90, elevation=90, focalpoint=(np.mean(qlims[0]), 0, 0), distance=1.8*np.sum(np.abs(qlims[0])))

    # Set-up user keyboard interactions
    qinit = np.copy(q)
    disturbance = 0
    reset = False
    original_keyPressEvent = fig.scene._vtk_control.keyPressEvent
    def keyPressEvent(event):
        global q, t, disturbance, reset, time_origin
        k = str(event.text())
        if k == '.':
            disturbance += 0.5
        elif k == ',':
            disturbance -= 0.5
        elif k == ' ':
            disturbance = 0
        elif k == 'r':
            q = np.copy(qinit)
            t = 0
            time_origin = time.time()
            disturbance = 0
            reset = True
        # else:
        #     original_keyPressEvent(event)
        #     print "Got unknown keypress: ", k
    fig.scene._vtk_control.keyPressEvent = keyPressEvent
    print "Increment / decrement disturbance force with '>' / '<' and cancel disturbance with ' '."
    print "Reset simulation with 'r'.\n"

    # Simulation loop function
    time_origin = time.time()
    @mlab.animate(delay=50)  # 20 FPS is best Mayavi can do
    def simulate():
        global q, t, reset
        while True:

            # Simulate physics up to realtime
            while t < time.time() - time_origin and not reset:

                # Obtain control decision, enforce input limits, apply user's disturbance
                u = np.clip(opt_control(t), ulims[0], ulims[1]) + disturbance

                # Verlet-integrate state vector forward one dt, enforce state constraints
                qdot = f(q, u)
                pose_next = q[:n_d] + dt*qdot[:n_d] + 0.5*dt**2*qdot[n_d:]
                verlet = f(np.append(pose_next, q[n_d:]), u)[n_d:]
                twist_next = q[n_d:] + dt*(qdot[n_d:] + verlet)/2
                q = np.append(pose_next, twist_next)
                q[0] = np.clip(q[0], qlims[0][0], qlims[0][1])
                # q = opt_state(t)  # for visualizing perfect execution # ???
                t += dt

            # Update animation
            reset = False
            cart.mlab_source.set(x=q[0])
            pole.mlab_source.set(x=(q[0], q[0]+l*np.sin(q[1])), z=(0, -l*np.cos(q[1])))
            disp.text = str(np.round(t, 1))
            yield

    # Begin visualization
    simulate()
    mlab.show()  # blocking

else:

    # Timeseries storage
    T = np.arange(0, 40, dt)
    U = np.zeros(len(T))
    Q = np.zeros((len(T), n_q))
    Q[0] = q

    # Simulation loop
    for i, t in enumerate(T[:-1]):

        # Obtain control decision, enforce input limits
        U[i] = np.clip(opt_control(t), ulims[0], ulims[1])

        # Verlet-integrate state vector forward one dt, enforce state constraints
        qdot = f(Q[i], U[i])
        pose_next = Q[i, :n_d] + dt*qdot[:n_d] + 0.5*dt**2*qdot[n_d:]
        verlet = f(np.append(pose_next, Q[i, n_d:]), U[i])[n_d:]
        twist_next = Q[i, n_d:] + dt*(qdot[n_d:] + verlet)/2
        Q[i+1] = np.append(pose_next, twist_next)
        Q[i+1, 0] = np.clip(Q[i+1, 0], qlims[0][0], qlims[0][1])

    # Plotting
    print "State at t={}:\n{}".format(tN, Q[int(tN/dt)])
    pyplot.plot(T, Q[:, 0], label="pos")
    pyplot.plot(T, Q[:, 1], label="ang")
    pyplot.plot(T, Q[:, 2], label="vel")
    pyplot.plot(T, Q[:, 3], label="angvel")
    pyplot.plot(T, U, label="force")
    pyplot.xlim([T[0], T[-1]])
    pyplot.xlabel("Time", fontsize=16)
    pyplot.ylabel("Results", fontsize=16)
    pyplot.legend(fontsize=16)
    pyplot.grid(True)
    pyplot.show()  # blocking
