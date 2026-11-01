'''
Power-thresholded EEG phase estimation
======================================
'''

# %% [markdown]
# A practical guide to deciding when an EEG window has enough alpha-band
# activity to attempt phase estimation. This example uses ETP and the same
# resting-state recording as the ETP tutorial: subject 1 from OpenNeuro
# ``ds004504``.
#
# The power cutoff is estimated from windows in the training half of the
# recording, then evaluated on the held-out half. The 25th percentile below is
# only an example choice: the useful operating point depends on how much data
# you can reject and what phase-estimation performance your experiment needs.
# For context on EEG-defined excitability states and stimulation, see
# :cite:t:`Zrenner2018-hh`. This example derives its own cutoff instead of
# treating a study-specific power value as universal; power depends on the
# recording setup and signal scale.

# %%
from EEGPhasePy.estimators import ETP
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mne
import openneuro

sys.path.append("../../..")


# %% [markdown]
# Download and load the same eyes-closed resting-state recording used in the
# ETP example.

# %%
openneuro.download(
    dataset='ds004504',
    tag='1.0.8',
    target_dir='ds004504',
    include=['sub-001']
)

raw_rs = mne.io.read_raw_eeglab(
    'ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set',
    preload=True,
    verbose=False
)

fs_rs = int(raw_rs.info['sfreq'])
raw_rs.pick_channels(['O1'])
rs_signal = raw_rs.get_data()[0]


# %% [markdown]
# Use the first half to fit ETP and calibrate the threshold, and the first two
# minutes of the second half for held-out comparison. Keeping calibration and
# evaluation separate avoids choosing the threshold based on the results we
# report, while limiting the length keeps this offline tutorial responsive.

# %%
split = len(rs_signal) // 2
training_signal_rs = rs_signal[:split]
testing_signal_rs = rs_signal[split:split + 120 * fs_rs]

rt_filter_rs = signal.firwin(128, [8, 12], fs=fs_rs, pass_zero=False)
gt_filter_rs = signal.firwin(500, [8, 12], fs=fs_rs, pass_zero=False)

etp_rs = ETP(rt_filter_rs, gt_filter_rs, fs_rs, window_len=2000)
etp_rs.fit(training_signal_rs, min_ipi=int(fs_rs * 1/12))

window_len_rs = int(2 * fs_rs)
window_step_rs = int(0.5 * fs_rs)


# %% [markdown]
# ``check_power_threshold`` can return the measured power directly. It
# calculates mean squared Hilbert-envelope amplitude after applying the
# real-time filter and removing its edge samples. This is a signal-power
# measure, not a power spectral density. MNE returns EEG in volts, so the
# plotted power is converted to microvolts squared, while the cutoff passed
# for threshold checks remains in volts squared.


calibration_powers = []
calibration_step_rs = int(1.0 * fs_rs)
for start in range(
        0, len(training_signal_rs) - window_len_rs + 1,
        calibration_step_rs):
    window = training_signal_rs[start:start + window_len_rs]
    calibration_powers.append(etp_rs.check_power_threshold(
        window, return_power=True))

calibration_powers = np.asarray(calibration_powers)


# %% [markdown]
# Choose the cutoff according to the rejection rate that is practical for the
# experiment. Here we use the 25th percentile of training-window power, which
# means roughly one quarter of calibration windows fall below the cutoff.
# This is a starting point for the example, not a claim that 25% is optimal.
# In a real study, inspect this distribution, choose candidate cutoffs before
# final evaluation, and use an independent validation/test recording to assess
# the accuracy-versus-usable-data tradeoff.
#
# Calibration windows are sampled every second, so they overlap by 50%. The
# percentile describes windows, not independent observations or participants.

# %%
threshold_quantile = 0.25
power_threshold_rs = float(np.quantile(
    calibration_powers, threshold_quantile))
power_threshold_microvolt_squared = power_threshold_rs * 1e12

print(
    f"Example cutoff: {power_threshold_microvolt_squared:.2f} uV^2 "
    f"(training {threshold_quantile:.0%} power percentile)"
)
print(
    f"Calibration windows at or above cutoff: "
    f"{100 * np.mean(calibration_powers >= power_threshold_rs):.1f}%"
)

fig, ax = plt.subplots()
ax.hist(calibration_powers * 1e12, bins=50)
ax.axvline(
    power_threshold_microvolt_squared,
    color='red',
    linestyle='--',
    label='Selected cutoff'
)
ax.set(
    xlabel='Mean squared alpha envelope (microvolts squared)',
    ylabel='Number of overlapping windows',
    title='Training-window alpha power used to choose the cutoff'
)
ax.legend()


# %% [markdown]
# Run the same pseudo-real-time loop twice, advancing the 2-second window by
# 500 ms each time to keep this offline example responsive. Without a
# threshold, every window is offered to ETP. With a threshold, ``predict`` is
# called only after ``check_power_threshold`` passes. A passing window may
# still fail phase estimation if ETP cannot find a peak.

# %%
def run_etp_sim(power_threshold=None):
    window_i = 2 * fs_rs
    triggers = []
    windows_passing_power = 0
    windows_checked = 0

    while window_i + window_len_rs < len(testing_signal_rs):
        window = testing_signal_rs[window_i:window_i + window_len_rs]
        windows_checked += 1

        if power_threshold is not None and not etp_rs.check_power_threshold(
                window, power_threshold):
            window_i += window_step_rs
            continue

        windows_passing_power += 1

        try:
            samples_to_wait = etp_rs.predict(window, target_phase=0)
            trigger_sample = (
                window_i + window_len_rs + samples_to_wait)
            if 0 <= trigger_sample < len(testing_signal_rs):
                triggers.append(trigger_sample)
        except RuntimeError:
            pass  # ETP could not find a peak in this window

        window_i += window_step_rs

    if not triggers:
        raise RuntimeError(
            'No triggers were produced. Recheck the recording and estimator '
            'settings, or choose a less restrictive power threshold.')

    phase_mean = etp_rs.mean_phase_from_triggers(
        testing_signal_rs, triggers, degree=True)
    phase_std = etp_rs.std_phase_from_triggers(
        testing_signal_rs, triggers, degree=True)
    accuracy = etp_rs.phase_accuracy_from_triggers(
        testing_signal_rs, triggers, 0)

    return {
        'triggers': triggers,
        'windows_checked': windows_checked,
        'windows_passing_power': windows_passing_power,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'accuracy': accuracy,
    }


results_without_threshold = run_etp_sim()
results_with_threshold = run_etp_sim(power_threshold_rs)


# %% [markdown]
# Compare the number of phase-estimation attempts and triggers as well as
# circular phase statistics on the held-out data. The threshold may improve
# phase consistency, but it will generally reduce the number of eligible
# windows and triggers. Look at both outcomes when deciding whether the cutoff
# is useful for your experiment.

# %%
def print_results(label, results):
    passed = results['windows_passing_power']
    checked = results['windows_checked']
    print(f"\n{label}")
    print(f"Windows sent to phase estimator: {passed}/{checked} "
          f"({100 * passed / checked:.1f}%)")
    print(f"Triggers: {len(results['triggers'])}")
    print(f"Mean phase: {results['phase_mean']:.1f} degrees")
    print(f"Phase standard deviation: {results['phase_std']:.1f} degrees")
    print(f"Phase accuracy: {100 * results['accuracy']:.1f}%")


print_results('Without a power threshold', results_without_threshold)
print_results('With the example power threshold', results_with_threshold)

for label, results in (
        ('Without power threshold', results_without_threshold),
        ('With power threshold', results_with_threshold)):
    triggers = results['triggers']
    polar_hist_fig = etp_rs.polar_histogram_from_triggers(
        testing_signal_rs, triggers)
    polar_hist_fig.suptitle(label)
    waveform_fig = etp_rs.plot_mean_std_waveform_from_triggers(
        testing_signal_rs, triggers, tmin=0.1, tmax=0.1)
    waveform_fig.suptitle(label)


# %% [markdown]
# The same gate works with PHASTIMATE (the AR phase estimator): check the
# power first, then call ``predict`` only when the window passes. For example:
#
# .. code-block:: python
#
#    if phastimate.check_power_threshold(window, power_threshold):
#        if phastimate.predict(window, target_phase=0, tolerance=15):
#            triggers.append(window_end_sample)
#
# Use the estimator's same real-time filter, window duration, electrode,
# reference, and signal units when transferring a calibrated threshold. If
# these change, recalibrate the power distribution. See the power-threshold
# guide for the API details.
