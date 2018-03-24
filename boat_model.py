"""
Contains a model of a 3DOF marine ship. Supplies parameter values,
a function for the full nonlinear dynamic, and functions for
jacobians of said dynamic. Running this script as __main__ will
present a quick open-loop simulation, comparing the nonlinear
dynamics to the linearized (about initial condition) dynamics.

State is: [x_world_position (m),
           y_world_position (m),
           heading_angle_from_x (rad),
           x_body_velocity (m/s),
           y_body_velocity (m/s),
           yaw_rate (rad/s)]

Input is: [x_body_force (N),
           y_body_force (N),
           z_torque (N*m)]

See:
T. I. Fossen, Handbook of Marine Craft Hydrodynamics
and Motion Control. Wiley, 2011. Chapter 13.

"""
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt

################################################# PHYSICAL PARAMETERS

# Boat inertia and center of gravity
m = 1.5*180  # kg
Iz = m*(3**2)  # kg*m**2
xg = -0.1  # m

# Fluid inertial effects
wm_xu = -0.025*m  # kg          # These expressions are just an okay starting point
wm_yv = -0.25*m  # kg           # if all you somewhat know are m, Iz, and xg
wm_yr = -0.25*m*xg  # kg*m
wm_nr = -0.25*Iz  # kg*m**2

# Drag
d_xuu = 0.25 * wm_xu  # N/(m/s)**2
d_yvv = 0.25 * wm_yv  # N/(m/s)**2
d_nrr = 0.25 * (wm_nr + wm_yr)  # (N*m)/(rad/s)**2

# Cross-flow
d_yrr = 0.25 * wm_yr  # N/(rad/s)**2
d_yrv = 0.25 * wm_yr  # N/(m*rad/s**2)
d_yvr = 0.25 * wm_yv  # N/(m*rad/s**2)
d_nvv = 0.25 * d_yvv  # (N*m)/(m/s)**2
d_nrv = 0.25 * d_yrv  # (N*m)/(m*rad/s**2)
d_nvr = 0.25 * (wm_nr + wm_yv)  # (N*m)/(m*rad/s**2)

################################################# EQUATIONS OF MOTION

# Inertial matrix, independent of state
M = np.array([
              [m - wm_xu,            0,            0],
              [        0,    m - wm_yv, m*xg - wm_yr],
              [        0, m*xg - wm_yr,   Iz - wm_nr]
            ])
Minv = npl.inv(M)


def f(q, u):
    """
    qdot = f(q, u)

    """
    # Centripetal-coriolis matrix
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

    # M*vdot + C*v + D*v = u  and  pdot = R*v
    return np.concatenate((R.dot(q[3:]), Minv.dot(u - (C + D).dot(q[3:]))))


def A(q):
    """
    Jacobian of f with respect to q.

    """
    return np.array([
                     [ 0, 0, - q[4]*np.cos(q[2]) - q[3]*np.sin(q[2]),                                                                                                                                                                                                                                             np.cos(q[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         -np.sin(q[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0],
                     [ 0, 0,   q[3]*np.cos(q[2]) - q[4]*np.sin(q[2]),                                                                                                                                                                                                                                             np.sin(q[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          np.cos(q[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0],
                     [ 0, 0,                         0,                                                                                                                                                                                                                                                   0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1],
                     [ 0, 0,                         0,                                                                                                                                                                                                     (d_xuu*abs(q[3]) + d_xuu*q[3]*np.sign(q[3]))/(m - wm_xu),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (m*q[5] - q[5]*wm_yv)/(m - wm_xu),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            (m*q[4] - 2*q[5]*wm_yr - q[4]*wm_yv + 2*m*q[5]*xg)/(m - wm_xu)],
                     [ 0, 0,                         0, -(q[5]*wm_yr**2 + m**2*q[5]*xg**2 - Iz*m*q[5] + Iz*q[5]*wm_xu + m*q[5]*wm_nr - q[5]*wm_nr*wm_xu - q[4]*wm_yr*wm_xu + q[4]*wm_yr*wm_yv - 2*m*q[5]*wm_yr*xg + m*q[4]*wm_xu*xg - m*q[4]*wm_yv*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(d_nrv*wm_yr*abs(q[5]) - d_yrv*wm_nr*abs(q[5]) + d_nvv*wm_yr*abs(q[4]) - d_yvv*wm_nr*abs(q[4]) - q[3]*wm_yr*wm_xu + q[3]*wm_yr*wm_yv + Iz*d_yrv*abs(q[5]) + Iz*d_yvv*abs(q[4]) + m*q[3]*wm_xu*xg - m*q[3]*wm_yv*xg + Iz*d_yvr*q[5]*np.sign(q[4]) + Iz*d_yvv*q[4]*np.sign(q[4]) - d_nrv*m*xg*abs(q[5]) - d_nvv*m*xg*abs(q[4]) + d_nvr*q[5]*wm_yr*np.sign(q[4]) - d_yvr*q[5]*wm_nr*np.sign(q[4]) + d_nvv*q[4]*wm_yr*np.sign(q[4]) - d_yvv*q[4]*wm_nr*np.sign(q[4]) - d_nvr*m*q[5]*xg*np.sign(q[4]) - d_nvv*m*q[4]*xg*np.sign(q[4]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(q[3]*wm_yr**2 + d_nrr*wm_yr*abs(q[5]) - d_yrr*wm_nr*abs(q[5]) + d_nvr*wm_yr*abs(q[4]) - d_yvr*wm_nr*abs(q[4]) + m**2*q[3]*xg**2 - Iz*m*q[3] + Iz*q[3]*wm_xu + m*q[3]*wm_nr - q[3]*wm_nr*wm_xu + Iz*d_yrr*abs(q[5]) + Iz*d_yvr*abs(q[4]) - 2*m*q[3]*wm_yr*xg + Iz*d_yrr*q[5]*np.sign(q[5]) + Iz*d_yrv*q[4]*np.sign(q[5]) - d_nrr*m*xg*abs(q[5]) - d_nvr*m*xg*abs(q[4]) + d_nrr*q[5]*wm_yr*np.sign(q[5]) - d_yrr*q[5]*wm_nr*np.sign(q[5]) + d_nrv*q[4]*wm_yr*np.sign(q[5]) - d_yrv*q[4]*wm_nr*np.sign(q[5]) - d_nrr*m*q[5]*xg*np.sign(q[5]) - d_nrv*m*q[4]*xg*np.sign(q[5]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)],
                     [ 0, 0,                         0,                                             (q[4]*wm_yv**2 + m*q[4]*wm_xu - m*q[4]*wm_yv - q[5]*wm_yr*wm_xu + q[5]*wm_yr*wm_yv - q[4]*wm_xu*wm_yv + m*q[5]*wm_xu*xg - m*q[5]*wm_yv*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),                (q[3]*wm_yv**2 - d_nrv*m*abs(q[5]) - d_nvv*m*abs(q[4]) + d_nrv*wm_yv*abs(q[5]) - d_yrv*wm_yr*abs(q[5]) + d_nvv*wm_yv*abs(q[4]) - d_yvv*wm_yr*abs(q[4]) + m*q[3]*wm_xu - m*q[3]*wm_yv - q[3]*wm_xu*wm_yv + d_yrv*m*xg*abs(q[5]) + d_yvv*m*xg*abs(q[4]) - d_nvr*m*q[5]*np.sign(q[4]) - d_nvv*m*q[4]*np.sign(q[4]) + d_nvr*q[5]*wm_yv*np.sign(q[4]) - d_yvr*q[5]*wm_yr*np.sign(q[4]) + d_nvv*q[4]*wm_yv*np.sign(q[4]) - d_yvv*q[4]*wm_yr*np.sign(q[4]) + d_yvr*m*q[5]*xg*np.sign(q[4]) + d_yvv*m*q[4]*xg*np.sign(q[4]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),                                      -(d_nrr*m*abs(q[5]) + d_nvr*m*abs(q[4]) - d_nrr*wm_yv*abs(q[5]) + d_yrr*wm_yr*abs(q[5]) - d_nvr*wm_yv*abs(q[4]) + d_yvr*wm_yr*abs(q[4]) + q[3]*wm_yr*wm_xu - q[3]*wm_yr*wm_yv - m*q[3]*wm_xu*xg + m*q[3]*wm_yv*xg - d_yrr*m*xg*abs(q[5]) - d_yvr*m*xg*abs(q[4]) + d_nrr*m*q[5]*np.sign(q[5]) + d_nrv*m*q[4]*np.sign(q[5]) - d_nrr*q[5]*wm_yv*np.sign(q[5]) + d_yrr*q[5]*wm_yr*np.sign(q[5]) - d_nrv*q[4]*wm_yv*np.sign(q[5]) + d_yrv*q[4]*wm_yr*np.sign(q[5]) - d_yrr*m*q[5]*xg*np.sign(q[5]) - d_yrv*m*q[4]*xg*np.sign(q[5]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)]
                   ])


# Jacobian of f with respect to u, independent of state
B = np.array([ 
              [             0,                                                                                             0,                                                                                             0],
              [             0,                                                                                             0,                                                                                             0],
              [             0,                                                                                             0,                                                                                             0],
              [ 1/(m - wm_xu),                                                                                             0,                                                                                             0],
              [             0,   -(Iz - wm_nr)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(wm_yr - m*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)],
              [             0, -(wm_yr - m*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),    -(m - wm_yv)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)]
            ])


# Returns an angle on [-pi, pi]
def unwrap(ang):
    return np.mod(ang+np.pi, 2*np.pi) - np.pi

# Adds a perturbation to a state, q [+] dq
def qplus(q, dq):
    qp = q + dq
    qp[2] = unwrap(qp[2])
    return qp

# Subtracts two states, ql [-] qr
def qminus(ql, qr):
    dq = ql - qr
    dq[2] = unwrap(dq[2])
    return dq

################################################# SIMULATION

if __name__ == "__main__":

    # Simulation duration and timestep
    T = 20  # s
    dt = 0.005  # s

    # Initial condition
    q0 = np.array([-5, 10, -0.1, 0.2, -0.1, 0.1], dtype=np.float64) # [m, m, rad, m/s, m/s, rad/s]
    q = np.copy(q0)
    qlin = np.copy(q0)
    u = np.array([0, 0, 0], dtype=np.float64)  # [N, N, N*m]

    # Define time domain
    t_arr = np.arange(0, T, dt)

    # Preallocate results memory
    q_history = np.zeros((len(t_arr), len(q)))
    qlin_history = np.zeros((len(t_arr), len(qlin)))
    u_history = np.zeros((len(t_arr), int(len(q)/2)))

    # Integrate dynamics using first-order forward stepping
    for i, t in enumerate(t_arr):

        # Some exogenous input
        u = 300*np.sin(0.5*t) * np.array([1, 1, 2])

        # Record this instant
        q_history[i] = q
        qlin_history[i] = qlin
        u_history[i] = u

        # Step forward, qnext = qlast + qdot*dt
        q = qplus(q, f(q, u)*dt)
        qlin = qplus(qlin, (A(q0).dot(qminus(qlin, q0)) + B.dot(u))*dt)

    # Figure for individual results
    fig1 = plt.figure()
    fig1.suptitle('State Evolution', fontsize=20)
    fig1rows = 2
    fig1cols = 4

    # Plot x position
    ax = fig1.add_subplot(fig1rows, fig1cols, 1)
    ax.set_title('X Position (m)', fontsize=16)
    ax.plot(t_arr, q_history[:, 0], 'g', label="nonlinear")
    ax.plot(t_arr, qlin_history[:, 0], 'k', label="linearized")
    ax.grid(True)
    ax.legend()

    # Plot y position
    ax = fig1.add_subplot(fig1rows, fig1cols, 2)
    ax.set_title('Y Position (m)', fontsize=16)
    ax.plot(t_arr, q_history[:, 1], 'g',
             t_arr, qlin_history[:, 1], 'k')
    ax.grid(True)

    # Plot yaw position
    ax = fig1.add_subplot(fig1rows, fig1cols, 3)
    ax.set_title('Heading (deg)', fontsize=16)
    ax.plot(t_arr, np.rad2deg(q_history[:, 2]), 'g',
             t_arr, np.rad2deg(qlin_history[:, 2]), 'k')
    ax.grid(True)

    # Plot control efforts
    ax = fig1.add_subplot(fig1rows, fig1cols, 4)
    ax.set_title('Wrench (N, N, N*m)', fontsize=16)
    ax.plot(t_arr, u_history[:, 0], 'b',
             t_arr, u_history[:, 1], 'g',
             t_arr, u_history[:, 2], 'r')
    ax.grid(True)

    # Plot x velocity
    ax = fig1.add_subplot(fig1rows, fig1cols, 5)
    ax.set_title('Surge (m/s)', fontsize=16)
    ax.plot(t_arr, q_history[:, 3], 'g',
             t_arr, qlin_history[:, 3], 'k')
    ax.set_xlabel('Time (s)')
    ax.grid(True)

    # Plot y velocity
    ax = fig1.add_subplot(fig1rows, fig1cols, 6)
    ax.set_title('Sway (m/s)', fontsize=16)
    ax.plot(t_arr, q_history[:, 4], 'g',
             t_arr, qlin_history[:, 4], 'k')
    ax.set_xlabel('Time (s)')
    ax.grid(True)

    # Plot yaw velocity
    ax = fig1.add_subplot(fig1rows, fig1cols, 7)
    ax.set_title('Yaw (deg/s)', fontsize=16)
    ax.plot(t_arr, np.rad2deg(q_history[:, 5]), 'g',
             t_arr, np.rad2deg(qlin_history[:, 5]), 'k')
    ax.set_xlabel('Time (s)')
    ax.grid(True)

    # Plot norm linearization errors
    ax = fig1.add_subplot(fig1rows, fig1cols, 8)
    ax.set_title('Norm Linearization Error', fontsize=16)
    ax.plot(t_arr, npl.norm([qminus(q_history[i], qlin_history[i]) for i in xrange(len(t_arr))], axis=1), 'k')
    ax.set_xlabel('Time (s)')
    ax.grid(True)

    plt.show()
