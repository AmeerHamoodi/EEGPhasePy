'''
Alpha phase estimation using EPN
===================================
'''

# %% [markdown]
# This tutorial walks through EEG phase estimation in the alpha band using the EEG Phase Prediction Network (EPN) :cite:t:`Liu2025-epn`.
# If you use EPN in your research, please cite :cite:t:`Liu2025-epn`.
#
# Two examples are provided: one using a simulated alpha signal and one using real resting state EEG from OpenNeuro dataset ds004504 (eyes-closed resting state, 500 Hz).
# Installation of mne and openneuro are necessary for this tutorial.

# %%
from EEGPhasePy.estimators import EPN
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mne
import openneuro

sys.path.append("../../..")


# %% [markdown]
# Simulated EEG
# -------------
# Generate the simulated training and testing EEG data. We use a sampling rate of 500 Hz since that is what EPN was designed for.

# %%
fs = 500
time_data = np.arange(0, 200, 1/fs)

training_signal_clean = np.sin(2 * np.pi * 10 * time_data)
training_signal = training_signal_clean + \
    np.random.normal(0, 2, len(time_data))

testing_signal_clean = np.sin(2 * np.pi * 10 * time_data)
testing_signal = testing_signal_clean + np.random.normal(0, 2, len(time_data))

# %% [markdown]
# Define the real-time and ground-truth alpha bandpass filters.
#
# EPN doesn't filter the window before estimating phase, so the real-time filter is only used if you check the power of a window with `check_power_threshold`.
# The ground truth filter is much more important for EPN than it is for the other estimators. It is used to create the phase EPN is trained to estimate, so the network
# will only ever be as good as the ground truth it learns from. It is also used to extract the phase for phase stats (i.e. accuracy, mean and std of phase) as well as
# to generate the polar histogram and trigger waveform plots.

# %%
rt_filter = signal.firwin(64, [8, 12], fs=fs, pass_zero=False)
gt_filter = signal.firwin(151, [8, 12], fs=fs, pass_zero=False)

# %% [markdown]
# Construct the EPN class and fit it to the training data. The default parameters match the network described in :cite:t:`Liu2025-epn`: a 500 ms window,
# an average pooling layer of 26 ms (13 samples at 500 Hz), 500 hidden units and a dropout probability of 0.9.
#
# Fitting takes a training window every 10 ms of the training data and trains the network to output the ground truth phase at the end of each window.
# We pass `random_state` so that the results are reproducible.

# %%
epn = EPN(rt_filter, gt_filter, fs)
epn.fit(training_signal, random_state=0)

# %% [markdown]
# The training loss for each epoch is stored in `loss_history`. If the loss is still dropping at the end of training, you can increase the number of epochs with
# the `epochs` argument of `fit`. The loss won't reach 0 since dropout is applied during training and the noise in the signal limits how well the phase can be estimated.

# %%
fig, ax = plt.subplots()
ax.plot(np.arange(1, len(epn.loss_history) + 1), epn.loss_history)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (MSE)")
ax.set_title("EPN training loss")
plt.show()

# %%


def run_epn_sim(target_phase, tolerance, accuracy_target, tmin=0.1, tmax=0.1):
    window_i = 2*fs
    window_len = int(0.5*fs)
    window_step = int(0.002*fs)  # using a step of 2 ms

    triggers = []

    while window_i + window_len < len(testing_signal):
        window = testing_signal[window_i:window_i + window_len]

        # the third parameter in `epn.predict` indicates the tolerance in degrees
        if epn.predict(window, target_phase, tolerance):
            triggers.append(window_i + window_len)
            # skip ahead so that triggers aren't delivered back-to-back
            window_i += window_len

        window_i += window_step

    polar_hist_fig = epn.polar_histogram_from_triggers(
        testing_signal, triggers)
    waveform_fig = epn.plot_mean_std_waveform_from_triggers(
        testing_signal, triggers, tmin=tmin, tmax=tmax)

    phase_standard_deviation = epn.std_phase_from_triggers(
        testing_signal, triggers, degree=True)
    phase_mean = epn.mean_phase_from_triggers(
        testing_signal, triggers, degree=True)
    accuracy = epn.phase_accuracy_from_triggers(
        testing_signal, triggers, accuracy_target)

    print("Mean phase: " + str(phase_mean))
    print("Std phase: " + str(phase_standard_deviation))
    print("Accuracy: " + str(100*accuracy))

    return polar_hist_fig, waveform_fig


# %% [markdown]
# All estimators in EEGPhasePy contain a ``predict`` method that takes the unfiltered sliding window data as an argument. Like PHASTIMATE, EPN estimates the
# phase at the current time, so ``predict`` returns ``True`` when the current phase matches the target phase within the given tolerance. The phase EPN estimates
# is the phase at the sample immediately after the window, so the trigger sample is ``window_i + window_len``.
#
# Since EPN only tells us whether the current phase is the target phase, it needs to be run often. Here we pass in a new window every 2 ms, which is the same as an
# amplifier sending a packet at 500 Hz. We target alpha peaks first (``target_phase = 0``).

# %%
polar_hist_fig, waveform_fig = run_epn_sim(
    target_phase=0, tolerance=10, accuracy_target=0)

# %% [markdown]
# Different phases can be targeted by changing the ``target_phase`` argument (in degrees). Below we target the alpha rising phase (``target_phase = 270``).

# %%
polar_hist_fig, waveform_fig = run_epn_sim(
    target_phase=270, tolerance=10, accuracy_target=270, tmin=0.05, tmax=0.05)

# %% [markdown]
# Similarly, to target troughs we change ``target_phase`` to ``180``.

# %%
polar_hist_fig, waveform_fig = run_epn_sim(
    target_phase=180, tolerance=10, accuracy_target=180, tmin=0.05, tmax=0.05)

# %% [markdown]
# You may notice the mean phase sits a few degrees below the target. ``predict`` returns ``True`` the first time the estimated phase enters the tolerance window,
# and since phase increases over time this tends to happen just before the target phase. Lowering the tolerance reduces this bias, at the cost of fewer windows
# meeting the criteria.

# %% [markdown]
# Resting State EEG (OpenNeuro ds004504)
# ---------------------------------------
#
# Here we run EPN on real resting state EEG from the OpenNeuro dataset
# ``ds004504`` — *"A dataset of EEG recordings from: Alzheimer's disease,
# Frontotemporal dementia and Healthy subjects"*. We will use the eyes-closed resting state recording from subject 1 (sub-001).
#
# The recording is ~10 minutes long at 500 Hz, which is the same sampling rate used by :cite:t:`Liu2025-epn`.
#
# We download only subject 1's files using ``openneuro-py``, then load the
# EEGLAB ``.set`` file with MNE.

# %%
openneuro.download(
    dataset='ds004504',
    tag='1.0.8',
    target_dir='ds004504',
    include=['sub-001']
)

# %%
raw_rs = mne.io.read_raw_eeglab(
    'ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set',
    preload=True,
    verbose=False
)

fs_rs = int(raw_rs.info['sfreq'])  # 500 Hz

# We will perform estimation from O1 since it tends to have prominent alpha activity
raw_rs.pick_channels(['O1'])
rs_signal = raw_rs.get_data()[0]

# %% [markdown]
# Define the alpha bandpass filters for 500 Hz. Since we have access to the full recording when applying the ground truth filter, we can use a high order filter.
#
# :cite:t:`Liu2025-epn` applied a Hjorth (surface Laplacian) spatial filter around their target electrode before EPN. EEGPhasePy estimators work on a single
# channel, so if you would like to do the same, apply the spatial filter to your data before passing it into `fit` and `predict`. To keep this example simple
# we use O1 on its own.

# %%
rt_filter_rs = signal.firwin(64, [8, 12], fs=fs_rs, pass_zero=False)
gt_filter_rs = signal.firwin(500, [8, 12], fs=fs_rs, pass_zero=False)

# %% [markdown]
# With ~10 minutes of recording we split the data in half: the first half is used to train EPN and the second half is used for the pseudo-real-time
# simulation. :cite:t:`Liu2025-epn` trained EPN on 4 minutes of resting state EEG for each participant.

# %%
split = len(rs_signal) // 2
training_signal_rs = rs_signal[:split]
testing_signal_rs = rs_signal[split:]

epn_rs = EPN(rt_filter_rs, gt_filter_rs, fs_rs)
epn_rs.fit(training_signal_rs, random_state=0)

# %% [markdown]
# Run the pseudo-real-time simulation on the held-out testing signal, targeting alpha peaks (phase = 0) with a new window every 2 ms.

# %%
window_i_rs = 2 * fs_rs
window_len_rs = int(0.5 * fs_rs)
window_step_rs = int(0.002 * fs_rs)

triggers_rs = []

while window_i_rs + window_len_rs < len(testing_signal_rs):
    window = testing_signal_rs[window_i_rs:window_i_rs + window_len_rs]

    if epn_rs.predict(window, 0, 10):
        triggers_rs.append(window_i_rs + window_len_rs)
        window_i_rs += window_len_rs

    window_i_rs += window_step_rs

# %%
polar_hist_fig_rs = epn_rs.polar_histogram_from_triggers(
    testing_signal_rs, triggers_rs)
waveform_fig_rs = epn_rs.plot_mean_std_waveform_from_triggers(
    testing_signal_rs, triggers_rs, tmin=0.1, tmax=0.1)

phase_std_rs = epn_rs.std_phase_from_triggers(
    testing_signal_rs, triggers_rs, degree=True)
phase_mean_rs = epn_rs.mean_phase_from_triggers(
    testing_signal_rs, triggers_rs, degree=True)
accuracy_rs = epn_rs.phase_accuracy_from_triggers(
    testing_signal_rs, triggers_rs, 0)

print("Mean phase: " + str(phase_mean_rs))
print("Std phase: " + str(phase_std_rs))
print("Accuracy: " + str(100 * accuracy_rs))

# %% [markdown]
# The mean phase should be close to our target phase. A standard deviation between 50-70 degrees appears to be normal based on a range of published literature.
# Accuracy should be interpreted keeping in mind that 50% indicates completely random phase locking, 0% indicates anti-phase locking and 100% is perfect phase locking.
