#!/usr/bin/env python2
"""
This Python script reads the result.txt file written by the
solve executable and visualizes the data for review.

Dependencies: NumPy, matplotlib

Some good alternatives to matplotlib for visualization
are Mayavi, PyGame, and OpenCV.

"""
from __future__ import division
import time
import numpy as np
from matplotlib import pyplot, cm

# Extract and parse the data
print("Parsing the data...")
data = np.genfromtxt("result.txt", delimiter=',', dtype=np.float64)
positions = np.linspace(0, 1, data.shape[1]-1)
times = data[:, 0]
temperatures = data[:, 1:]
time_step = times[1] - times[0]  # assumes constant temporal discretization

# Create figure
fig = pyplot.figure()
axes1 = fig.add_subplot(2, 1, 1)
axes2 = fig.add_subplot(2, 1, 2)
axes1.set_xticklabels([])
axes1.set_yticklabels([])
axes2.set_xlim([0, 1])
axes2.set_xlabel("position")
axes2.set_ylabel("temperature")
axes2.grid(True)
pyplot.show(block=False)

# Some display parameters
rod_diameter = 50  # pixels
time_per_disp = 0.01  # seconds
iter_per_disp = int(time_per_disp / time_step)

# Initial temperature image and plot
image = axes1.imshow(np.tile(temperatures[0], (rod_diameter, 1)), cmap=cm.gist_heat)
curve = axes2.plot(positions, temperatures[0], linewidth=3)[0]

# Data-point index and realtime origin
index = 0
start = time.time()

# Animate
print("Visualizing (terminate program to quit)...")
while True:
    # While simulation time is behind realtime
    while times[index] < time.time() - start:
        # Display current time
        fig.suptitle("t = {}".format(np.round(times[index], 2)), fontsize=16)
        # Update temperature image
        image.set_array(np.tile(temperatures[index], (rod_diameter, 1)))
        # Update temperature plot
        curve.set_ydata(temperatures[index])
        # Update figure and move to next time data-point
        fig.canvas.draw()
        index += iter_per_disp
        # Repeat when done with data
        if(index >= len(times)):
            index = 0
            start = time.time()
    # Else rest and check again
    time.sleep(1e-5)
