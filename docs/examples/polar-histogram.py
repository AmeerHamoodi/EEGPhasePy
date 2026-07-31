'''
Polar histograms
=================
'''


# %%
import EEGPhasePy.viz as viz
import sys
import numpy as np
import scipy.signal as signal

sys.path.append("../../..")


# %%
fs = 2000
time_data = np.arange(0, 200, 1/fs)

# here we are generating some simulated oscillatory data to plot
signal_clean = np.sin(2 * np.pi * 10 * time_data)

# %% [markdown]
# Here, we identify the location of the peaks in our signal and add some random gaussian noise to the phase data of our peaks to simulate the results of real phase estimation. The `viz.plot_polar_histogram` method requires only that the phase data be passed into it

# %%
peaks = signal.find_peaks(signal_clean)[0]
phase_data = np.angle(signal.hilbert(signal_clean))

polar_hist = viz.plot_polar_histogram(
    # we add some noise to the phase data by randomly sampling from a normal distribution centered at 0 with a standard deviation of 0.7
    phase_data[peaks] + np.random.normal(0, 0.7, len(peaks)))
