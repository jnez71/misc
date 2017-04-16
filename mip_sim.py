"""
Simulation of planar mobile inverted pendulum (MIP) robot,
assuming no-slip rigid body dynamics.

Semantics:
    cm == center of mass
    cw == center of wheel
    cs == center of sensors
    q  == full state
    u  == motor torque effort about -z
    w  == wheel angle about -z (redundant state)

Coordinate systems:
    world frame with origin in wheel at t=0:
        x horizontal (not incline), y vertical, z = x cross y
    incline frame with origin in wheel at t=0:
        x along ground, y normal to ground, z = x cross y
    body frame with origin in wheel:
        x = y cross z, y from cw to cm, z = z_world

Full state is [m, m/s, rad, rad/s]:
    position of cw along incline-frame x direction
    velocity of cw along incline-frame x direction
    orientation angle from world y to body y
    angular velocity of orientation angle

"""

################################################# SETUP

# Dependencies
from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Simulation parameters
duration = 20  # s
timestep = 0.005  # s
framerate = 30  # FPS

# Pendulum parameters
cw_to_cm = np.array([0, 2])  # m, body y
mass_pend = 1  # kg
inertia_pend = mass_pend*cw_to_cm[1]**2  # kg*m^2
friction_pend = 0.2  # (N*m)/(rad/s)

# Wheel parameters
radius = 0.5  # m
mass_wheel = 1  # kg
inertia_wheel = 0.5*mass_wheel*radius**2  # kg*m^2
friction_wheel = 1.2  # (N*m)/(rad/s)

# World parameters
gravity = np.array([0, -9.81])  # m/s^2, world y
incline = np.deg2rad(0)  # rad
david = None  # fuck
david_shift = 0 # you

# Sensor parameters
encoder_tpr = 1024  # tic/rev
gyro_bias = np.deg2rad(2)  # rad/s
gyro_stdv = np.deg2rad(2)  # rad/s
inclinometer_bias = np.deg2rad(1)  # rad
inclinometer_stdv = np.deg2rad(5)  # rad

# Actuator parameters
torque_limit = 20  # N*m
torque_deltamax = 0.1*torque_limit / timestep  # N*m/s

# Control system initialization
controller_active = True
kp = 50  # N*m/rad
kd = 2*np.sqrt(kp)  # N*m/(rad/s)
max_mobile_tilt = np.deg2rad(15)  # rad
desired_position = 4  # m, None => no preference
last_encoder = 0

# Initial condition
q = np.array([0, 0, np.deg2rad(2), 0])
w = q[0]/radius
u = 0
p = 0

def dynamics(q, u, p):
    """
    Returns state derivative qdot.
    Takes current state q, motor input torque u, and disturbance torque p.
    See <http://renaissance.ucsd.edu/courses/mae143c/MIPdynamics.pdf> (rederived with incline).

    """
    # Angle of pendulum in incline frame
    ang = q[2] - incline

    # Mass matrix
    M = np.array([
                  [(mass_wheel + mass_pend)*radius**2 + inertia_wheel, mass_pend*radius*cw_to_cm[1]*np.cos(ang)],
                  [mass_pend*radius*cw_to_cm[1]*np.cos(ang), inertia_pend + mass_pend*cw_to_cm[1]**2]
                ])

    # Gravity effect
    g = np.array([
                  -mass_pend*radius*cw_to_cm[1]*q[3]**2*np.sin(ang) + mass_wheel*radius*gravity[1]*np.sin(incline),
                  mass_pend*gravity[1]*cw_to_cm[1]*np.sin(q[2])
                ])

    # Friction force
    d = np.array([
                  -friction_wheel * (q[1] + np.arctan(q[1])),
                  friction_pend * q[3]
                ])

    # Dynamics
    accel_wheel_neg, accel_pend = npl.inv(M).dot(np.array([-u, p+u]) - g - d)

    return np.array([q[1], -accel_wheel_neg*radius, q[3], accel_pend])

################################################# SIMULATION

# Define time domain
t_arr = np.arange(0, duration, timestep)

# Preallocate results memory
q_history = np.zeros((len(t_arr), 4))
q_ref_history = np.zeros((len(t_arr), 4))
q_meas_history = np.zeros((len(t_arr), 4))
u_history = np.zeros(len(t_arr))
w_history = np.zeros(len(t_arr))
p_history = np.zeros(len(t_arr))
incline_history = np.zeros(len(t_arr))

# Integrate dynamics using first-order forward stepping
for i, t in enumerate(t_arr):

    # Sensor measurements
    encoder_meas = (np.round((w/(2*np.pi))*encoder_tpr)/encoder_tpr)*2*np.pi * radius  # m
    deriv_est = (encoder_meas - last_encoder) / timestep  # m/s
    last_encoder = encoder_meas
    inclinometer_meas = q[2] + inclinometer_bias + np.random.normal(0, inclinometer_stdv)  # rad
    gyro_meas = q[3] + gyro_bias + np.random.normal(0, gyro_stdv)  # rad/s

    # Controller's decision
    if controller_active and q[2] > -np.pi/2 and q[2] < np.pi/2:
        if desired_position is None:
            mobile_tilt = 0
        else:
            mobile_tilt = np.clip(0.1*(encoder_meas-desired_position) + 0.1*q[1], -max_mobile_tilt, max_mobile_tilt)
        u_ref = kp*(mobile_tilt - (inclinometer_meas-inclinometer_bias)) + kd*(0 - (gyro_meas-gyro_bias))
    else:
        mobile_tilt = 0
        u_ref = 0

    # Actuator slew and saturation
    if u < u_ref:
        u = np.clip(u + torque_deltamax*timestep, u, u_ref)
    elif u > u_ref:
        u = np.clip(u - torque_deltamax*timestep, u_ref, u)
    u = np.clip(u, -torque_limit, torque_limit)

    # External disturbance
    if t < max(t_arr)/2:
        p = 0#-5*np.sin(2*t)*np.cos(0.2*t)
    else:
        p = 0
    
    # Record this instant
    q_history[i, :] = q
    q_ref_history[i, :] = [desired_position, 0, mobile_tilt, 0]
    q_meas_history[i, :] = [encoder_meas, deriv_est, inclinometer_meas, gyro_meas]
    u_history[i] = u
    w_history[i] = w
    p_history[i] = p
    incline_history[i] = incline

    # First-order integrate qdot = f(q, u, p)
    q = q + dynamics(q, u, p)*timestep
    w = q[0]/radius

    # Update incline for david
    if david is not None:
        incline = np.arctan(2*david*(q[0] - david_shift))

################################################# VISUALIZATION

# Plots
fig1 = plt.figure()
fig1.suptitle('Results', fontsize=20)

ax1 = fig1.add_subplot(2, 3, 1)
ax1.set_ylabel('Wheel Position (m)', fontsize=16)
ax1.plot(t_arr, q_history[:, 0], 'k',
         t_arr, q_ref_history[:, 0], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 2)
ax1.set_ylabel('Pendulum Angle (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 2]), 'k',
         t_arr, np.rad2deg(q_ref_history[:, 2]), 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 3)
ax1.set_ylabel('Input Torque (N*m)', fontsize=16)
ax1.plot(t_arr, u_history, 'k',
         t_arr, p_history, 'r--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 4)
ax1.set_ylabel('Wheel Velocity (m/s)', fontsize=16)
ax1.plot(t_arr, q_history[:, 1], 'k',
         t_arr, q_ref_history[:, 1], 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 5)
ax1.set_ylabel('Pendulum Velocity (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 3]), 'k',
         t_arr, np.rad2deg(q_ref_history[:, 3]), 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 6)
ax1.set_ylabel('State Estimation Errors', fontsize=16)
ax1.plot(t_arr, q_history[:, 0] - q_meas_history[:, 0], 'k', label='position')
ax1.plot(t_arr, q_history[:, 1] - q_meas_history[:, 1], 'r', label='velocity')
ax1.plot(t_arr, q_history[:, 2] - q_meas_history[:, 2], 'g', label='pend angle')
ax1.plot(t_arr, q_history[:, 3] - q_meas_history[:, 3], 'b', label='pend rate')
ax1.legend()
ax1.set_xlabel('Time (s)')
ax1.grid(True)

print("Close the plot window to continue to animation.")
plt.show()

# Animation
fig2 = plt.figure()
fig2.suptitle('Evolution', fontsize=24)
plt.axis('equal')

ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xlabel('- World X (m)+')
ax2.set_ylabel('- World Y (m)+')
ax2.grid(True)

# (break q[0] incline coordinate into world coordinates)
x_history = q_history[:, 0] * np.cos(incline_history)
y_history = q_history[:, 0] * np.sin(incline_history)

ax_lim = 2*npl.norm(cw_to_cm) + np.ceil(radius)
ax2.set_xlim([-ax_lim, ax_lim])
ax2.set_ylim([-ax_lim, ax_lim])

floor_lim = np.max(np.abs(q_history[:, 0])) * 2
pscale = 0.25

graphic_floor = ax2.plot([-floor_lim*np.cos(incline_history[0]) + radius*np.sin(incline_history[0]), floor_lim*np.cos(incline_history[0]) + radius*np.sin(incline_history[0])], [-floor_lim*np.sin(incline_history[0])-radius*np.cos(incline_history[0]), floor_lim*np.sin(incline_history[0])-radius*np.cos(incline_history[0])], color='g', linewidth=5)
graphic_wheel = ax2.add_patch(plt.Circle((x_history[0], y_history[0]), radius=radius, fc='k'))
graphic_ind = ax2.plot([x_history[0], x_history[0] + radius*np.sin(w_history[0])],
                       [y_history[0], y_history[0] + radius*np.cos(w_history[0])], color='y', linewidth=3)
graphic_pend = ax2.plot([x_history[0], x_history[0] - cw_to_cm[1]*np.sin(q_history[0, 2])],
                        [y_history[0], y_history[0] + cw_to_cm[1]*np.cos(q_history[0, 2])], color='b', linewidth=5)
graphic_dist = ax2.plot([x_history[0] - cw_to_cm[1]*np.sin(q_history[0, 2]), x_history[0] - cw_to_cm[1]*np.sin(q_history[0, 2]) - pscale*p_history[0]*np.cos(q_history[0, 2])],
                        [y_history[0] + cw_to_cm[1]*np.cos(q_history[0, 2]), y_history[0] + cw_to_cm[1]*np.cos(q_history[0, 2]) - pscale*p_history[0]*np.sin(q_history[0, 2])], color='r', linewidth=3)

def ani_update(arg, ii=[0]):

    i = ii[0]  # don't ask...

    if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
        fig2.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

    graphic_floor[0].set_data([-floor_lim*np.cos(incline_history[i]) + radius*np.sin(incline_history[i]), floor_lim*np.cos(incline_history[i]) + radius*np.sin(incline_history[i])], [-floor_lim*np.sin(incline_history[i])-radius*np.cos(incline_history[i]), floor_lim*np.sin(incline_history[i])-radius*np.cos(incline_history[i])])
    graphic_wheel.center = (x_history[i], y_history[i])
    graphic_ind[0].set_data([x_history[i], x_history[i] + radius*np.sin(w_history[i])],
                            [y_history[i], y_history[i] + radius*np.cos(w_history[i])])
    graphic_pend[0].set_data([x_history[i], x_history[i] - cw_to_cm[1]*np.sin(q_history[i, 2])],
                             [y_history[i], y_history[i] + cw_to_cm[1]*np.cos(q_history[i, 2])])
    graphic_dist[0].set_data([x_history[i] - cw_to_cm[1]*np.sin(q_history[i, 2]), x_history[i] - cw_to_cm[1]*np.sin(q_history[i, 2]) - pscale*p_history[i]*np.cos(q_history[i, 2])],
                             [y_history[i] + cw_to_cm[1]*np.cos(q_history[i, 2]), y_history[i] + cw_to_cm[1]*np.cos(q_history[i, 2]) - pscale*p_history[i]*np.sin(q_history[i, 2])])

    ii[0] += int(1 / (timestep * framerate))
    if ii[0] >= len(t_arr):
        print("Resetting animation!")
        ii[0] = 0

    return [graphic_floor, graphic_wheel, graphic_ind, graphic_pend, graphic_dist]

# Run animation
print("Starting animation.\nBlack: wheel, Blue: pendulum, Yellow: angle indicator, Red: disturbance, Green: ground.")
animation = ani.FuncAnimation(fig2, func=ani_update, interval=timestep*1000)
plt.show()
