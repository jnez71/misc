"""
Compare linearized dynamics to nonlinear dynamics
for Fossen's boat model.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

################################################# PHYSICAL PARAMETERS

# Simulation duration, timestep, and animation
T = 20  # s
dt = 0.001  # s
framerate = 60  # fps

# Initial condition
q = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)  # [m, m, rad, m/s, m/s, rad/s]
qlin = np.copy(q)
u = np.array([0, 0, 0], dtype=np.float64)  # [N, N, N*m]

# Boat inertia and center of gravity
m = 1000  # kg
Iz = 1500  # kg*m**2
xg = 0.1  # m

# Fluid inertial effects
wm_xu = -0.025*m  # kg
wm_yv = -0.25*m  # kg
wm_yr = -0.25*m*xg  # kg*m
wm_nr = -0.25*Iz  # kg*m**2

# Drag
d_xuu = 0.25 * wm_xu  # N/(m/s)**2
d_yvv = 0.25 * wm_yv # N/(m/s)**2
d_nrr = 0.25 * (wm_nr + wm_yr) # (N*m)/(rad/s)**2

# Cross-flow
d_yrr = 0.25 * wm_yr # N/(rad/s)**2
d_yrv = 0.25 * wm_yr  # N/(m*rad/s**2)
d_yvr = 0.25 * wm_yv  # N/(m*rad/s**2)
d_nvv = 0.25 * d_yvv # (N*m)/(m/s)**2
d_nrv = 0.25 * d_yrv # (N*m)/(m*rad/s**2)
d_nvr = 0.25 * (wm_nr + wm_yv) # (N*m)/(m*rad/s**2)

################################################# EQUATIONS OF MOTION

def nonlinear_dynamics(q, u):
    """
    qdot = f(q, u)

    """
    # Externally set parameters
    global m, Iz, xg, wm_xu, wm_yv, wm_yr, wm_nr,\
           d_xuu, d_yvv, d_nrr, d_yrr, d_yrv, d_yvr,\
           d_nvv, d_nrv, d_nvr

    # Unpack for keeping MatLab compatability during debug
    px = q[0]
    py = q[1]
    pr = q[2]
    vx = q[3]
    vy = q[4]
    vn = q[5]

    ux = u[0]
    uy = u[1]
    uz = u[2]

    # Mass matrix
    M = np.array([
                  [m - wm_xu,            0,            0],
                  [        0,    m - wm_yv, m*xg - wm_yr],
                  [        0, m*xg - wm_yr,   Iz - wm_nr]
                ])

    # Centripetal coriolis matrix
    C = np.array([
                  [                                     0,                0, (wm_yr - m*xg)*q[5] + (wm_yv - m)*q[4]],
                  [                                     0,                0,                       (m - wm_xu)*q[3]],
                  [(m*xg - wm_yr)*q[5] + (m - wm_yv)*q[4], (wm_xu - m)*q[3],                                      0]
                ])

    # Drag matrix
    D = np.array([
                  [-d_xuu*abs(q[3]),                                    0,                                    0],
                  [               0, -(d_yvv*abs(q[4]) + d_yrv*abs(q[5])), -(d_yvr*abs(q[4]) + d_yrr*abs(q[5]))],
                  [               0, -(d_nvv*abs(q[4]) + d_nrv*abs(q[5])), -(d_nvr*abs(q[4]) + d_nrr*abs(q[5]))]
                ])

    # Rotation matrix (orientation, converts body to world)
    R = np.array([
                  [np.cos(q[2]), -np.sin(q[2]), 0],
                  [np.sin(q[2]),  np.cos(q[2]), 0],
                  [           0,             0, 1]
                ])

    # M*vdot + C*v + D*v = u  and  etadot = R*v
    return np.concatenate((R.dot(q[3:]), np.linalg.inv(M).dot(u - (C + D).dot(q[3:]))))


def linearized_dynamics(q, u):
    """
    qdot = Aq + Bu
    A = jac(f, q)
    B = jac(f, u)

    """
    # Externally set parameters
    global m, Iz, xg, wm_xu, wm_yv, wm_yr, wm_nr,\
           d_xuu, d_yvv, d_nrr, d_yrr, d_yrv, d_yvr,\
           d_nvv, d_nrv, d_nvr

    # Unpack for keeping MatLab compatability during debug
    px = q[0]
    py = q[1]
    pr = q[2]
    vx = q[3]
    vy = q[4]
    vn = q[5]

    ux = u[0]
    uy = u[1]
    uz = u[2]

    # Linearization cacophony
    qEQ = np.zeros_like(q)
    uEQ = np.zeros_like(u)

    A = np.array([
                  [ 0, 0, - vy*np.cos(pr) - vx*np.sin(pr),                                                                                                                                                                                                                                             np.cos(pr),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         -np.sin(pr),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0],
                  [ 0, 0,   vx*np.cos(pr) - vy*np.sin(pr),                                                                                                                                                                                                                                             np.sin(pr),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          np.cos(pr),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0],
                  [ 0, 0,                         0,                                                                                                                                                                                                                                                   0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1],
                  [ 0, 0,                         0,                                                                                                                                                                                                     (d_xuu*abs(vx) + d_xuu*vx*np.sign(vx))/(m - wm_xu),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (m*vn - vn*wm_yv)/(m - wm_xu),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            (m*vy - 2*vn*wm_yr - vy*wm_yv + 2*m*vn*xg)/(m - wm_xu)],
                  [ 0, 0,                         0, -(vn*wm_yr**2 + m**2*vn*xg**2 - Iz*m*vn + Iz*vn*wm_xu + m*vn*wm_nr - vn*wm_nr*wm_xu - vy*wm_yr*wm_xu + vy*wm_yr*wm_yv - 2*m*vn*wm_yr*xg + m*vy*wm_xu*xg - m*vy*wm_yv*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(d_nrv*wm_yr*abs(vn) - d_yrv*wm_nr*abs(vn) + d_nvv*wm_yr*abs(vy) - d_yvv*wm_nr*abs(vy) - vx*wm_yr*wm_xu + vx*wm_yr*wm_yv + Iz*d_yrv*abs(vn) + Iz*d_yvv*abs(vy) + m*vx*wm_xu*xg - m*vx*wm_yv*xg + Iz*d_yvr*vn*np.sign(vy) + Iz*d_yvv*vy*np.sign(vy) - d_nrv*m*xg*abs(vn) - d_nvv*m*xg*abs(vy) + d_nvr*vn*wm_yr*np.sign(vy) - d_yvr*vn*wm_nr*np.sign(vy) + d_nvv*vy*wm_yr*np.sign(vy) - d_yvv*vy*wm_nr*np.sign(vy) - d_nvr*m*vn*xg*np.sign(vy) - d_nvv*m*vy*xg*np.sign(vy))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(vx*wm_yr**2 + d_nrr*wm_yr*abs(vn) - d_yrr*wm_nr*abs(vn) + d_nvr*wm_yr*abs(vy) - d_yvr*wm_nr*abs(vy) + m**2*vx*xg**2 - Iz*m*vx + Iz*vx*wm_xu + m*vx*wm_nr - vx*wm_nr*wm_xu + Iz*d_yrr*abs(vn) + Iz*d_yvr*abs(vy) - 2*m*vx*wm_yr*xg + Iz*d_yrr*vn*np.sign(vn) + Iz*d_yrv*vy*np.sign(vn) - d_nrr*m*xg*abs(vn) - d_nvr*m*xg*abs(vy) + d_nrr*vn*wm_yr*np.sign(vn) - d_yrr*vn*wm_nr*np.sign(vn) + d_nrv*vy*wm_yr*np.sign(vn) - d_yrv*vy*wm_nr*np.sign(vn) - d_nrr*m*vn*xg*np.sign(vn) - d_nrv*m*vy*xg*np.sign(vn))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)],
                  [ 0, 0,                         0,                                             (vy*wm_yv**2 + m*vy*wm_xu - m*vy*wm_yv - vn*wm_yr*wm_xu + vn*wm_yr*wm_yv - vy*wm_xu*wm_yv + m*vn*wm_xu*xg - m*vn*wm_yv*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),                (vx*wm_yv**2 - d_nrv*m*abs(vn) - d_nvv*m*abs(vy) + d_nrv*wm_yv*abs(vn) - d_yrv*wm_yr*abs(vn) + d_nvv*wm_yv*abs(vy) - d_yvv*wm_yr*abs(vy) + m*vx*wm_xu - m*vx*wm_yv - vx*wm_xu*wm_yv + d_yrv*m*xg*abs(vn) + d_yvv*m*xg*abs(vy) - d_nvr*m*vn*np.sign(vy) - d_nvv*m*vy*np.sign(vy) + d_nvr*vn*wm_yv*np.sign(vy) - d_yvr*vn*wm_yr*np.sign(vy) + d_nvv*vy*wm_yv*np.sign(vy) - d_yvv*vy*wm_yr*np.sign(vy) + d_yvr*m*vn*xg*np.sign(vy) + d_yvv*m*vy*xg*np.sign(vy))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),                                      -(d_nrr*m*abs(vn) + d_nvr*m*abs(vy) - d_nrr*wm_yv*abs(vn) + d_yrr*wm_yr*abs(vn) - d_nvr*wm_yv*abs(vy) + d_yvr*wm_yr*abs(vy) + vx*wm_yr*wm_xu - vx*wm_yr*wm_yv - m*vx*wm_xu*xg + m*vx*wm_yv*xg - d_yrr*m*xg*abs(vn) - d_yvr*m*xg*abs(vy) + d_nrr*m*vn*np.sign(vn) + d_nrv*m*vy*np.sign(vn) - d_nrr*vn*wm_yv*np.sign(vn) + d_yrr*vn*wm_yr*np.sign(vn) - d_nrv*vy*wm_yv*np.sign(vn) + d_yrv*vy*wm_yr*np.sign(vn) - d_yrr*m*vn*xg*np.sign(vn) - d_yrv*m*vy*xg*np.sign(vn))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)]
                ])

    B = np.array([ 
                  [             0,                                                                                             0,                                                                                             0],
                  [             0,                                                                                             0,                                                                                             0],
                  [             0,                                                                                             0,                                                                                             0],
                  [ 1/(m - wm_xu),                                                                                             0,                                                                                             0],
                  [             0,   -(Iz - wm_nr)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(wm_yr - m*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)],
                  [             0, -(wm_yr - m*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),    -(m - wm_yv)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)]
                ])

    return A.dot(q - qEQ) + B.dot(u - uEQ) # + something?

################################################# SIMULATION

# Define time domain
t_arr = np.arange(0, T, dt)

# Preallocate results memory
q_history = np.zeros((len(t_arr), len(q)))
qlin_history = np.zeros((len(t_arr), len(qlin)))
u_history = np.zeros((len(t_arr), int(len(q)/2)))

# Integrate dynamics using first-order forward stepping
for i, t in enumerate(t_arr):

    # Some exogenous input
    u = 500*np.sin(0.1*t) * np.ones(3)

    # Record this instant
    q_history[i, :] = q
    qlin_history[i, :] = qlin
    u_history[i, :] = u

    # Step forward, qnext = qlast + qdot*dt
    q = q + nonlinear_dynamics(q, u)*dt
    qlin = qlin + linearized_dynamics(qlin, u)*dt

################################################# VISUALIZATION

# Figure for individual results
fig1 = plt.figure()
fig1.suptitle('State Evolution', fontsize=20)
fig1rows = 2
fig1cols = 4

# Plot x position
ax1 = fig1.add_subplot(fig1rows, fig1cols, 1)
ax1.set_title('X Position (m)', fontsize=16)
ax1.plot(t_arr, q_history[:, 0], 'g',
         t_arr, qlin_history[:, 0], 'k')
ax1.grid(True)

# Plot y position
ax1 = fig1.add_subplot(fig1rows, fig1cols, 2)
ax1.set_title('Y Position (m)', fontsize=16)
ax1.plot(t_arr, q_history[:, 1], 'g',
         t_arr, qlin_history[:, 1], 'k')
ax1.grid(True)

# Plot yaw position
ax1 = fig1.add_subplot(fig1rows, fig1cols, 3)
ax1.set_title('Heading (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 2]), 'g',
         t_arr, np.rad2deg(qlin_history[:, 2]), 'k')
ax1.grid(True)

# Plot control efforts
ax1 = fig1.add_subplot(fig1rows, fig1cols, 4)
ax1.set_title('Wrench (N, N, N*m)', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'b',
         t_arr, u_history[:, 1], 'g',
         t_arr, u_history[:, 2], 'r')
ax1.grid(True)

# Plot x velocity
ax1 = fig1.add_subplot(fig1rows, fig1cols, 5)
ax1.set_title('Surge (m/s)', fontsize=16)
ax1.plot(t_arr, q_history[:, 3], 'g',
         t_arr, qlin_history[:, 3], 'k')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot y velocity
ax1 = fig1.add_subplot(fig1rows, fig1cols, 6)
ax1.set_title('Sway (m/s)', fontsize=16)
ax1.plot(t_arr, q_history[:, 4], 'g',
         t_arr, qlin_history[:, 4], 'k')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot yaw velocity
ax1 = fig1.add_subplot(fig1rows, fig1cols, 7)
ax1.set_title('Yaw (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 5]), 'g',
         t_arr, np.rad2deg(qlin_history[:, 5]), 'k')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot norm linearization errors
ax1 = fig1.add_subplot(fig1rows, fig1cols, 8)
ax1.set_title('Norm Linearization Error', fontsize=16)
ax1.plot(t_arr, npl.norm(q_history - qlin_history, axis=1), 'k')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

plt.show()
