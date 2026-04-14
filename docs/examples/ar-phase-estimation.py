'''
Alpha phase estimation using PHASTIMATE
================================================
'''


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
time_data = np.arange(0, 200, 1/fs)

signal_clean = np.sin(2 * np.pi * 10 * time_data)
signal_noisy = signal_clean + np.random.normal(0, 2, len(time_data))

# %% [markdown]
# Define the real-time and ground-truth alpha bandpass filters.
#
# The real-time filter should be lower order since it will be applied to a window with fewer samples than the ground truth filter. The ground truth filter is what will be used to extract the phase for phase stats (i.e. accuracy, mean and std of phase) as well as to generate the polar histogram and trigger waveform plots

# %%
rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)

# %% [markdown]
# Next, we will instantiate PHASTIMATE.
#
# Unlike ETP, PHASTIMATE doesn't necessarily require training data. With the right parameters
# it can work out of the box. However, you may benefit from optimizing the `window_edge` and `ar_order` parameters in the class, especially
# when there is a variability in SNR across participants.

# %%
phastimate = PHASTIMATE(rt_filter, gt_filter, fs, window_edge=35, ar_order=15)

# %% [markdown]
# Now, we will run a pseudo-real-time simulation using the testing signal. Here, we use the `predict` method of the `phastimate`
# object. `predict` works by returning a boolean value about whether the current phase is your target phase. A tolerance value may also be
# passed in. By default tolerance is 5 degrees. Tolerance indicates how close the current phase must be to the target phase for `predict`
# to return `True`. If you are finding that the time between phases being identified is too long you can try increasing tolerance. It is important to note that higher tolerance values will naturally reduce accuracy of this model.
# Additionally, you may not want to use this model if your amplifier has a low packet send rate. The original paper implementing AR phase estimation in real-time had a packet send rate of 500 Hz. At lower rates
# the model has fewer attempts to detect the target phase and as a result the experimenter will have less control over the inter-stimulus interval.

# %%


def run_phastimate_sim(target_phase, tolerance, accuracy_target, tmin=0.1, tmax=0.1):
    window_i = 2*fs
    window_len = int(0.5*fs)
    window_step = int(0.06*fs)  # using a step of 60 ms

    triggers = []

    while window_i + window_len < len(signal_noisy):
        window = signal_noisy[window_i:window_i + window_len]

        # the third parameter in `phastimate.predict` indicates the tolerance in degrees
        if phastimate.predict(window, target_phase, tolerance):
            triggers.append(window_i + window_len)

        window_i += window_step

    polar_hist_fig = phastimate.polar_histogram_from_triggers(
        signal_noisy, triggers)
    waveform_fig = phastimate.plot_mean_std_waveform_from_triggers(
        signal_noisy, triggers, tmin=tmin, tmax=tmax)

    phase_standard_deviation = phastimate.std_phase_from_triggers(
        signal_noisy, triggers, degree=True)
    phase_mean = phastimate.mean_phase_from_triggers(
        signal_noisy, triggers, degree=True)
    accuracy = phastimate.phase_accuracy_from_triggers(
        signal_noisy, triggers, accuracy_target)

    print("Mean phase: " + str(phase_mean))
    print("Std phase: " + str(phase_standard_deviation))
    print("Accuracy: " + str(100*accuracy))

    return polar_hist_fig, waveform_fig


# %% [markdown]
# All estimators in EEGPhasePy contain a ``predict`` method that takes the unfiltered
# sliding window data as an argument. With PHASTIMATE, ``predict`` returns ``True``
# when the current phase matches the target phase within the given tolerance. We target
# alpha peaks first (``target_phase = 0``).

# %%
polar_hist_fig, waveform_fig = run_phastimate_sim(
    target_phase=0, tolerance=15, accuracy_target=0)

# %% [markdown]
# As with ETP, PHASTIMATE can also quite easily detect different phases simply by
# changing the ``target_phase`` argument (which uses degrees). Below we target the
# alpha rising phase (``target_phase = 270``).

# %%
polar_hist_fig, waveform_fig = run_phastimate_sim(
    target_phase=270, tolerance=2, accuracy_target=270, tmin=0.05, tmax=0.05)

# %% [markdown]
# Similarly, to target troughs we change ``target_phase`` to ``180``.

# %%
polar_hist_fig, waveform_fig = run_phastimate_sim(
    target_phase=180, tolerance=5, accuracy_target=180, tmin=0.05, tmax=0.05)


# %% [markdown]
# The mean phase should be within 5deg of our target phase. A standard deviation between 50-70 degrees appears to be normal based on a range of published literature. Accuracy should be interepreted keeping in mind that 50% indicates completely random phase locking, 0% indicates anti-phase locking and 100% is perfect phase locking.
