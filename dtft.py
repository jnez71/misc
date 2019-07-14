#!/usr/bin/env python3
"""
Using a typical FFT routine and showing the principle
behind the DTFT computation.

"""
import numpy as np
from matplotlib import pyplot

##################################################

# Efficient practical usage
def fft(values, dt):
    freqs = np.fft.rfftfreq(len(values), dt)
    coeffs = np.sqrt(2/len(values)) * np.fft.rfft(values)  # scaled for unitarity
    coeffs[0] /= np.sqrt(2)  # don't "double count" the DC alias
    return (freqs, coeffs)

# Working principle
def dtft(values, dt):
    times = dt * np.arange(len(values))
    nyquist = 1/(2*dt)
    dw = 1/(dt*len(values))
    freqs = np.arange(0.0, nyquist+dw, dw)
    #                         (rad/s)/Hz      all w*t products
    dtft_matrix = np.exp(-1j * (2*np.pi) * np.outer(freqs, times))
    coeffs = np.sqrt(2/len(values)) * dtft_matrix.dot(values)  # scaled for unitarity
    coeffs[0] /= np.sqrt(2)  # don't "double count" the DC alias
    return (freqs, coeffs)

##################################################

def function(time):
    w = 20*np.pi
    value = 0.0
    for k in range(5):
        value += (k+1)*np.cos((k*w)*time)
    return value

dt = 0.001
times = np.arange(0.0, 0.2, dt)
values = function(times)

##################################################

fft_freqs, fft_coeffs = fft(values, dt)
dtft_freqs, dtft_coeffs = dtft(values, dt)

assert np.allclose(fft_freqs, dtft_freqs)
assert np.allclose(fft_coeffs, dtft_coeffs)

##################################################

# Demonstrate Parseval's theorem
print(np.linalg.norm(values))
print(np.linalg.norm(dtft_coeffs))

##################################################

fig = pyplot.figure()

ax = fig.add_subplot(2, 1, 1)
ax.plot(times, values)
ax.set_xlabel("Time (s)", fontsize=16)
ax.grid(True)

ax = fig.add_subplot(2, 1, 2)
ax.scatter(dtft_freqs, np.abs(dtft_coeffs))
ax.set_xlabel("Freq (Hz)", fontsize=16)
ax.grid(True)

pyplot.show()
