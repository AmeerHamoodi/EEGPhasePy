'''
Bayesian optimization of PHASTIMATE parameters
================================================
'''

# %% [markdown]
# It may be ideal in certain circumstances to systematically optimize the parameters involved in the PHASTIMATE toolbox. For instance, you may be dealing with a population known to contain a large degree of inter-individual variability in the SNR of your target frequency band. Thus, the PHASTIMATE parameters that work for one individual may not work well for another. You can optptimize the `window_edge` and `ar_order` parameters of PHASTIMATE to tailor the algorithm on an inter-individual basis.
#
# A brief explanation of each parameter: `window_edge` is the number of milliseconds of signal to remove after bandpass filtering. If you are using higher order filters, there will be greater edge effects and as a result more signal will need to be removed, hence a larger `window_edge` would be necessary,
# :cite:t:`Zrenner2020-zb` showed that adjusting `window_edge` can improve performance of EEG phase estimation on a participant-by-participant basis, which is why the study implemented genetic optimization of `window_edge`.
# In addition to `window_edge`, `ar_order` is also optimized for. `ar_order` indicates the number of samples the Yule-Walker autoregression should use to forward forecast the next 128ms of data. Higher `ar_order` increases the "historical dependence" of autoregression as more previous points are taken into account. Importantly, higher `ar_order` is more computationally expensive and as a result it does slow down model performance, but it also can increase accuracy of phase estimation Zrenner2020-zb
#
# Below we showcase how this can be done in EEGPhasePy using Bayesian optimization. If you use this algorithm please cite the BayesianOptimization library (https://github.com/bayesian-optimization/BayesianOptimization)
#
# The primary difference between genetic optimization and bayesian optimization is that this form of optimization does appear to result in slightly better optimization than genetic optimization. It is included to provide the user with options
# around which method of optimization they should use for AR phase estimation.  Please see https://github.com/bayesian-optimization/BayesianOptimization for information on how Bayesian optmization works.

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
#  .. note::
#    Filter order of `rt_filter` significantly affect the accuracy of phase estimation done in real-time or pseudo-real time. Higher order filters result in greater edge effects (resulting in less of the signal being usable for autoregression) but can result in much cleaner signals. It's important to balance between the two, which is why optimization of parameters on a participant-by-participant basis can be important in some cases.
#    The filter order of `gt_filter` is important in offline cases (i.e. when analyzing previously collected phase-triggered data).

# %% [markdown]
# Next, we will instantiate PHASTIMATE.

# %%
phastimate = PHASTIMATE(rt_filter, gt_filter, fs)

# %% [markdown]
# Now we will run the bayesian optimization algorithm using the `optimize_parameters` method with the `method` argument being set to bayesian. This method will update the `window_edge` and `ar_order` that result in the phase estimation accuracy

# All PHASTIMATE optimizations have a `window_edge` range of: 5ms to min(60ms, `window_len` / 8) and an `ar_order` range of: 1 to 0.1 * `sampling_rate`.

# %% [markdown]
# .. note::
#    Keep in mind that running these optimizations is very time consuming. It can take over 10 minutes to run the optimization.

# %%
phastimate.optimize_parameters(training_signal, 'bayesian')

# After completing the optimization, the `window_edge` and `ar_order` are automatically updated to the optimal values.

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

polar_hist_fig = phastimate.polar_histogram_from_triggers(
    testing_signal, triggers)
waveform_fig = phastimate.plot_mean_std_waveform_from_triggers(
    testing_signal, triggers, tmin=0.1, tmax=0.1)
