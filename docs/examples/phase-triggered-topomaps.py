'''
Offline C3 alpha phase-triggered topomaps
=========================================
'''

# %% [markdown]
# This example uses participant 1's eyes-closed resting recording (run 2)
# from the 64-channel PhysioNet EEG Motor Movement/Imagery dataset. It selects
# offline pseudo-trigger samples at the peak, falling slope, trough, and rising
# slope of C3 alpha activity, then plots the average whole-scalp alpha current
# source density (CSD) topography at each sample.
#
# Phase is estimated from a local surface-Laplacian approximation:
# ``C3 - mean(C1, C5, FC3, CP3)``. This spatial filter changes the C3 phase
# relative to using the raw C3 signal. For each topography, MNE's spherical
# spline CSD transform computes a surface Laplacian at every scalp channel.
# This emphasizes local spatial structure compared with plotting voltage.
# Because this is a retrospective resting recording, the example does not
# simulate real-time prediction and contains no TMS trials or responses.

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal as signal
from EEGPhasePy import viz


# %%
eeg_file = mne.datasets.eegbci.load_data(
    subjects=[1],
    runs=[2],
    update_path=True,
    verbose=False
)[0]
raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
mne.datasets.eegbci.standardize(raw)
raw.set_montage(mne.channels.make_standard_montage('standard_1005'))

sampling_rate = int(raw.info['sfreq'])
eeg_data = raw.get_data()
c3_index = raw.ch_names.index('C3')
neighbor_indices = [
    raw.ch_names.index(name) for name in ('C1', 'C5', 'FC3', 'CP3')
]
c3_laplacian = (
    eeg_data[c3_index] - eeg_data[neighbor_indices].mean(axis=0)
)


# %% [markdown]
# Filter the sensor data and the C3 Laplacian to the alpha band. The analytic
# phase of the Laplacian signal is used to choose the pseudo-trigger times.
# With this Hilbert phase convention, phase 0 is the positive peak,
# :math:`\\pi/2` is the falling slope, :math:`\\pi` is the trough, and
# :math:`-\\pi/2` is the rising slope. Filtering is zero-phase and uses the
# full recording, so this is an offline visualization rather than a real-time
# trigger simulation.

# %%
alpha_sos = signal.butter(
    4, [8, 12], btype='bandpass', fs=sampling_rate, output='sos')
c3_alpha_laplacian = signal.sosfiltfilt(alpha_sos, c3_laplacian)
c3_phase = np.angle(signal.hilbert(c3_alpha_laplacian))

# Compute a spherical-spline surface Laplacian independently at every
# electrode. Keep the local C3 Laplacian above for phase selection.
csd_raw = mne.preprocessing.compute_current_source_density(raw.copy())
csd_data = csd_raw.get_data()
csd_alpha = signal.sosfiltfilt(alpha_sos, csd_data, axis=-1)

phase_targets = {
    'Peak': 0.0,
    'Falling': np.pi / 2,
    'Trough': np.pi,
    'Rising': -np.pi / 2,
}

tolerance_degrees = 10
minimum_interval_seconds = 0.5
edge_samples = 2 * sampling_rate
valid_phase = c3_phase[edge_samples:-edge_samples]
trigger_samples_by_phase = {}

for phase_name, target_phase in phase_targets.items():
    phase_match = np.cos(valid_phase - target_phase)
    relative_samples, _ = signal.find_peaks(
        phase_match,
        height=np.cos(np.deg2rad(tolerance_degrees)),
        distance=int(minimum_interval_seconds * sampling_rate)
    )
    trigger_samples = relative_samples + edge_samples
    if len(trigger_samples) == 0:
        raise RuntimeError(
            f'No C3 phase matches found for the {phase_name.lower()} target')
    trigger_samples_by_phase[phase_name] = trigger_samples

    phase_error = np.angle(
        np.exp(1j * (c3_phase[trigger_samples] - target_phase)))
    mean_error_degrees = np.mean(np.abs(np.rad2deg(phase_error)))
    print(
        f'{phase_name}: {len(trigger_samples)} pseudo-triggers, '
        f'mean phase error {mean_error_degrees:.1f} degrees'
    )


# %% [markdown]
# Use the standard 1005 montage to position all 64 channels. Coordinates are
# centered and scaled for the plotting helper. MNE's CSD data are in V/m^2;
# convert to uV/cm^2 for readable colorbar values. Each map chooses its own
# symmetric color range; use the colorbar values, not color saturation alone,
# when comparing magnitudes across phases.

# %%
layout = mne.channels.make_eeg_layout(raw.info)
layout_positions = {
    channel_name: layout.pos[index, :2]
    for index, channel_name in enumerate(layout.names)
}
positions = np.array([layout_positions[name] for name in raw.ch_names])
positions -= positions.mean(axis=0)
positions /= np.max(np.linalg.norm(positions, axis=1))

csd_alpha_uv_cm2 = csd_alpha * 1e2
topomap_figures = {}

for phase_name, trigger_samples in trigger_samples_by_phase.items():
    trigger_csd = csd_alpha_uv_cm2[:, trigger_samples].T
    figure = viz.plot_topomap(trigger_csd, positions)
    figure.axes[0].set_title(
        f'C3 alpha {phase_name.lower()} '
        f'(n={len(trigger_samples)})'
    )
    figure.axes[-1].set_ylabel('Mean alpha CSD (uV/cm^2)')
    topomap_figures[phase_name] = figure

plt.show()
