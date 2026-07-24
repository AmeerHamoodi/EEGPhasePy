'''
Alpha phase estimation using ETP
===================================
'''

# %% [markdown]
# This tutorial willwalkthrough performing EEG phase estimation in the alpha band using ETP.
# Two examples are provided: one using a simulated alpha signal and one using real resting state EEG from OpenNeuro dataset ds004504 (eyes-closed resting state, 500 Hz).
# Installation of mne and openneuro are necessary for this tutorial.

# %%
from EEGPhasePy.estimators import ETP
import sys
import numpy as np
import scipy.signal as signal
import mne
import openneuro

sys.path.append("../../..")


# %% [markdown]
# Simulated EEG
# -------------
# Generate the simulated training and testing EEG data

# %%
fs = 2000
time_data = np.arange(0, 200, 1/fs)

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
# Construct the ETP class and fit it to the training data. We will use the default edge and window length for this example
#
# *A note on window length and edge effects:*
#
# When looking at other frequency bands that have more phase instability (as is the case with theta) or in cases of lower SNR, such as extracting phase data from the leisoned cortex of stroke patients, adjusting the window length and edge removed can significantly improve ETP performance. By increasing window length, more data is included in the sliding window allowing for a higher order filter to be used. This will help to more effectively extract the "signal" from "noise". However, increasing filter order will also increase the amount of data that is affected by filtering. Thus, you will need to increase the amount of signal removed from the edge of the window to compensate. This can potentially compromise ETP performance. It is important to balance all 3 of these factors

# %%
etp = ETP(rt_filter, gt_filter, fs)
etp.fit(training_signal, min_ipi=int(fs * 1/12))

# %%


def run_etp_sim(target_phase, accuracy_target, tmin=0.1, tmax=0.1):
    window_i = 2*fs
    window_len = int(0.5*fs)
    window_step = int(0.06*fs)  # using a step of 60 ms

    triggers = []

    while window_i + window_len < len(testing_signal):
        window = testing_signal[window_i:window_i + window_len]

        # predict returns samples to wait after the window end
        samples_to_wait = etp.predict(window, target_phase)
        trigger_sample = window_i + window_len + samples_to_wait
        if trigger_sample < len(testing_signal):
            triggers.append(trigger_sample)
        window_i += window_step

    polar_hist_fig = etp.polar_histogram_from_triggers(
        testing_signal, triggers)
    waveform_fig = etp.plot_mean_std_waveform_from_triggers(
        testing_signal, triggers, tmin=tmin, tmax=tmax)

    phase_standard_deviation = etp.std_phase_from_triggers(
        testing_signal, triggers, degree=True)
    phase_mean = etp.mean_phase_from_triggers(
        testing_signal, triggers, degree=True)
    accuracy = etp.phase_accuracy_from_triggers(
        testing_signal, triggers, accuracy_target)

    print("Mean phase: " + str(phase_mean))
    print("Std phase: " + str(phase_standard_deviation))
    print("Accuracy: " + str(100*accuracy))

    return polar_hist_fig, waveform_fig


# %% [markdown]
# Fitting the ETP model essentially calculates the optimal `Tadj` (inter-peak-interval), so we can proceed with
# pseudo-real-time simulations on the testing data. All estimators in EEGPhasePy
# contain a `predict` method that takes the unfiltered sliding window data as an
# argument. With ETP, `predict` returns the **number of samples to wait after the
# end of the window** before delivering the stimulus. In a real-time system, once
# the current data window has been received, the system waits the returned number
# of samples and then fires the stimulus. In simulation, the absolute trigger index
# is ``window_start + window_len + samples_to_wait``. We target alpha peaks
# first (``target_phase = 0``).

# %%
polar_hist_fig, waveform_fig = run_etp_sim(target_phase=0, accuracy_target=0)

# %% [markdown]
# The polar histogram has a bin size of 22 and indicates the number of triggers that occurred in each phase bin.
# In general, the more tightly clustered the polar histogram appears, the more consistently a phase is being targeted.
# Whereas, phase bin(s) in which the majority of triggers occur in, indicate the mean/mode of the phase being triggered at

# The waveform plot shows the average waveform with t = 0 indicating the time at which a pulse would occur at. The highlight around
# the waveform is the standard deviation at each timepoint with the upper bound of the highlight being +std and the lower bound being -std.

# %% [markdown]
# With ETP you can easily estimate any other phase by changing the `target_phase`
# argument. Below we target the alpha rising phase (:math:`3\pi/2`)

# %%
polar_hist_fig, waveform_fig = run_etp_sim(
    target_phase=270, accuracy_target=270, tmin=0.05, tmax=0.05)
# %% [markdown]
# Similarly, to target troughs we change `target_phase` to :math:`\pi`

# %%
polar_hist_fig, waveform_fig = run_etp_sim(
    target_phase=np.pi, accuracy_target=180, tmin=0.05, tmax=0.05)

# %% [markdown]
# Resting State EEG (OpenNeuro ds004504)
# ---------------------------------------
#
# Here we run ETP on real resting state EEG from the OpenNeuro dataset
# ``ds004504`` — *"A dataset of EEG recordings from: Alzheimer's disease,
# Frontotemporal dementia and Healthy subjects"*. We will use the eyes-closed resting state recording from subject 1 (sub-001).
#
# The recording is ~10 minutes long at 500 Hz.
#
# We download only subject 1's files using ``openneuro-py``, then load the
# EEGLAB ``.set`` file with MNE.

# %%
openneuro.download(
    dataset='ds004504',
    target_dir='ds004504',
    include=['sub-001']
)

# %% [markdown]
# After downloading the dataset, we will load a file in via mne's EEGLab reader

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
# Define alpha bandpass filters for 500 Hz. The 128 order real-time filter balances between extracted signal quality and
# edge effects caused by the filter. This often has to be played around with to optimize performance. We are planning to add the Bayesian Temporal Prediction
# algorithm that should fix this problem in the near future :cite:t:`Shirinpour2025`. For the "ground-truth" filter
# a higher-order filter since we have access to the full signal when applying that filter.

# %%
rt_filter_rs = signal.firwin(128, [8, 12], fs=fs_rs, pass_zero=False)
gt_filter_rs = signal.firwin(500, [8, 12], fs=fs_rs, pass_zero=False)

# %% [markdown]
# With ~10 minutes of recording we split the data in half: the first half is
# used to train ETP and the second half is used for the pseudo-real-time
# simulation. Importantly, ETP only requires 3 min of data for fitting

# %%
split = len(rs_signal) // 2
training_signal_rs = rs_signal[:split]
testing_signal_rs = rs_signal[split:]

etp_rs = ETP(rt_filter_rs, gt_filter_rs, fs_rs, window_len=2000)
etp_rs.fit(training_signal_rs, min_ipi=int(fs_rs * 1/12))

# %% [markdown]
# Run the pseudo-real-time simulation on the held-out testing signal,
# targeting alpha peaks (phase = 0).

# %%
window_i_rs = 2 * fs_rs
window_len_rs = int(2 * fs_rs)   # 1000 samples at 500 Hz
window_step_rs = int(0.06 * fs_rs)  # ~60 ms step

triggers_rs = []

while window_i_rs + window_len_rs < len(testing_signal_rs):
    window = testing_signal_rs[window_i_rs:window_i_rs + window_len_rs]

    try:
        # predict returns samples to wait after the window end;
        # adjust the second argument to change the desired phase
        samples_to_wait = etp_rs.predict(window, 0)
        trigger_sample = window_i_rs + window_len_rs + samples_to_wait
        if trigger_sample < len(testing_signal_rs):
            triggers_rs.append(trigger_sample)
    except RuntimeError:
        pass  # no peaks found in window, skip

    window_i_rs += window_step_rs

# %%
polar_hist_fig_rs = etp_rs.polar_histogram_from_triggers(
    testing_signal_rs, triggers_rs)
waveform_fig_rs = etp_rs.plot_mean_std_waveform_from_triggers(
    testing_signal_rs, triggers_rs, tmin=0.1, tmax=0.1)

phase_std_rs = etp_rs.std_phase_from_triggers(
    testing_signal_rs, triggers_rs, degree=True)
phase_mean_rs = etp_rs.mean_phase_from_triggers(
    testing_signal_rs, triggers_rs, degree=True)
accuracy_rs = etp_rs.phase_accuracy_from_triggers(
    testing_signal_rs, triggers_rs, 0)

print("Mean phase: " + str(phase_mean_rs))
print("Std phase: " + str(phase_std_rs))
print("Accuracy: " + str(100 * accuracy_rs))
