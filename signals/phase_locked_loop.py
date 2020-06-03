#!/usr/bin/env python3
"""
Simple phase-locked-loop experiment.
https://en.wikipedia.org/wiki/Phase-locked_loop#Time_domain_model

"""
# Dependencies
import numpy as np
from scipy.signal import butter, zpk2ss
from matplotlib import pyplot

# Display configuration
np.set_printoptions(suppress=True)
pyplot.rcParams["axes.grid"] = True
pyplot.rcParams["font.size"] = 20

##################################################

# Periodic 1-dimensional real-valued signal
class Signal:
    def __init__(self, amplitude, frequency, shift):
        self.amplitude = np.float(amplitude)
        self.frequency = np.float(frequency)
        self.shift = np.float(shift)

    def tap(self, time):
        return self.amplitude*np.sin((2*np.pi*self.frequency)*(time - self.shift))

##################################################

# Phase-locked-loop using modulation-based detector and linear filter
class PhaseLockedLoop:
    def __init__(self, A, b, c, freequency):
        self.A = np.array(np.atleast_2d(A), float)  # filter dynamic (n,n)
        self.b = np.array(b, float).reshape(len(self.A), 1)  # filter input transform (n,1)
        self.c = np.array(c, float).reshape(1, len(self.A))  # filter output transform (1,n)
        self.x = np.zeros((len(self.A), 1), float)  # filter state (n,1)
        self.freequency = np.float(freequency)  # free frequency of the controlled oscillator
        self.phase = np.float(0)  # phase estimate

    def seek(self, target, timestep):
        replica = np.sin(self.phase)  # internal oscillator output
        error = target*replica - 0.5  # modulate target with replica, cosine/2 of phase difference is in there
        self.x += timestep*(self.A.dot(self.x) + self.b.dot(error))  # filter out harmonics
        advance = self.c.dot(self.x)  # compute oscillator control as proportional to filtered error
        self.phase += timestep*(2*np.pi*(self.freequency + advance))  # advance the oscillator
        return replica, advance  # for records

##################################################

# Discrete Fourier transform normalized magnitude (for analysis)
def fourier(values, period):
    frequencies = np.fft.rfftfreq(len(values), period)
    coefficients = np.sqrt(2.0/len(values)) * np.fft.rfft(values)  # scaled for unitarity
    coefficients[0] /= np.sqrt(2.0)
    magnitudes = np.abs(coefficients) / np.linalg.norm(coefficients)
    return (frequencies, magnitudes)

##################################################

# Create simulation time domain
timestep = 0.0001
times = np.arange(0, 0.3, timestep)

# Design PLL using Butterworth filter
f = 100
n = 8
k = 20
A, b, c, _ = zpk2ss(*butter(n, 2*np.pi*f, btype="lowpass", analog=True, output="zpk"))
pll = PhaseLockedLoop(A=A, b=b, c=k*c, freequency=f)

# Run simulation
targets = Signal(1.5, f-3, 0.007).tap(times)
replicas, advances = np.transpose([pll.seek(target, timestep) for target in targets])

##################################################

# Plot time-domain results
figure, axes = pyplot.subplots(2, 1, sharex=True)
axes[0].plot(times, targets, color='r', label="Target")
axes[0].plot(times, replicas, color='k', label="PLL")
axes[0].set_xlim([0, times[-1]])
axes[0].legend()
axes[1].plot(times, advances, color='m', label="Advance")
axes[0].set_xlim([0, times[-1]])
axes[1].legend()
axes[-1].set_xlabel("Time")

# # Plot steady-state (assumed halfway) frequency-domain results
# figure, axes = pyplot.subplots(1, 1, sharex=True)
# axes.plot(*fourier(replicas[len(times)//2:], timestep), color='k', label="PLL")
# axes.plot(*fourier(advances[len(times)//2:], timestep), color='m', label="Advance")
# axes.set_xlim([0, 5*f])
# axes.legend()
# axes.set_ylabel("Normalized Magnitude")
# axes.set_xlabel("Frequency")

# Block to display visualization
print("Close plots to finish!")
pyplot.show()

##################################################
