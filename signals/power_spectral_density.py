#!/usr/bin/env python3
"""
Monte-carlo estimate of a simple power-spectral-density.

"""
import numpy as np
from matplotlib import pyplot

np.set_printoptions(suppress=True)
pyplot.rcParams["axes.grid"] = True
pyplot.rcParams["font.size"] = 20

##################################################

def bodemag(values, dt):
    freqs = np.fft.rfftfreq(len(values), dt)[:-1]
    coeffs = np.sqrt(2.0/len(values)) * np.fft.rfft(values)[:-1]
    return (freqs, 20*np.log10(np.abs(coeffs)))

##################################################

def sample(x0, a, r, n):
    X = np.zeros(n, float)
    X[0] = x0
    for i in range(n-1):
        X[i+1] = a*X[i] + np.random.normal(0.0, np.sqrt(r))
    return X

##################################################

print("Computing ensemble statistics...")
size = 5000
ensemble = np.array([sample(x0=1.0, a=0.8, r=0.001, n=200) for i in range(size)])
mean = np.mean(ensemble, axis=0)
variance = np.var(ensemble, axis=0)

print("Computing spectral statistics...")
spectra = []
for trajectory in ensemble:
    frequencies, amplitudes = bodemag(trajectory[len(trajectory)//2:], 0.0001)
    spectra.append(amplitudes)
spectra_mean = np.mean(spectra, axis=0)
spectra_variance = np.var(spectra, axis=0)

##################################################

print("Plotting...")
figure, axes = pyplot.subplots(3, 2)
axes[-1, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Ensemble\n(100/{0})".format(size))
for trajectory in ensemble[:100]:
    axes[0, 0].plot(trajectory)
axes[1, 0].set_ylabel("Mean")
axes[1, 0].plot(mean)
axes[2, 0].set_ylabel("Variance")
axes[2, 0].plot(variance)
axes[-1, 1].set_xlabel("Frequency")
for spectrum in spectra[:100]:
    axes[0, 1].semilogx(frequencies, spectrum)
axes[1, 1].semilogx(frequencies, spectra_mean)
axes[2, 1].semilogx(frequencies, spectra_variance)
print("Close plots to continue...")
pyplot.show()

##################################################
