'''
Bayesian optimization of PHASTIMATE parameters
================================================
'''

# %% [markdown]
# It may be ideal in certain circumstances to systematically optimize the parameters involved in the PHASTIMATE toolbox. For instance, you may be dealing with a population known to contain a large degree of inter-individual variability in the SNR of your target frequency band. Thus, the PHASTIMATE parameters that work for one individual may not work well for another. You can optptimize the `window_edge` and `ar_order` parameters of PHASTIMATE to tailor the algorithm on an inter-individual basis.
#
# Below we showcase how this can be done in EEGPhasePy using Bayesian optimization. If you use this algorithm please cite the BayesianOptimization library (https://github.com/bayesian-optimization/BayesianOptimization)

# %%
from EEGPhasePy.estimators import PHASTIMATE
import sys
import numpy as np
import scipy.signal as signal

sys.path.append("../../..")


# %% [markdown]
# Generate the simulated training and testing EEG data

# %%
fs = 2000
time_data = np.arange(0, 90, 1/fs)

training_signal_clean = np.sin(2 * np.pi * 10 * time_data)
training_signal = training_signal_clean + \
    np.random.normal(0, 2, len(time_data))

testing_signal_clean = np.sin(2 * np.pi * 10 * time_data)
testing_signal = testing_signal_clean + np.random.normal(0, 2, len(time_data))

# %% [markdown]
# Define the real-time and ground-truth alpha bandpass filters.
#
# The real-time filter should be lower order since it will be applied to a window with fewer samples than the ground truth filter. The ground truth filter is what will be used to extract the phase for phase stats (i.e. accuracy, mean and std of phase) as well as to generate the polar histogram and trigger waveform plots

# %%
rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)

# %% [markdown]
# Next, we will instantiate PHASTIMATE.

# %%
phastimate = PHASTIMATE(rt_filter, gt_filter, fs)

# %% [markdown]
# Now we will run the bayesian optimization algorithm using the `optimize_parameters` method with the `method` argument being set to bayesian. This method will update the `window_edge` and `ar_order` that result in the phase estimation accuracy

# %%
phastimate.optimize_parameters(training_signal, 'bayesian')

# %%
window_i = 2*fs
window_len = int(0.5*fs)
window_step = int(0.06*fs)  # using a step of 60 ms

triggers = []

while window_i + window_len < len(testing_signal):
    window = testing_signal[window_i:window_i + window_len]

    if phastimate.predict(window, 0, 15):
        triggers.append(window_i + window_len)

    window_i += window_step

# sphinx_gallery_thumbnail_number = 1
polar_hist_fig = phastimate.polar_histogram_from_triggers(
    testing_signal, triggers)
waveform_fig = phastimate.plot_mean_std_waveform_from_triggers(
    testing_signal, triggers, tmin=0.1, tmax=0.1)
