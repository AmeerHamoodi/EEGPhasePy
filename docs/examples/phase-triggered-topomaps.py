'''
Offline C3 alpha phase-triggered topomaps
=========================================
'''

# %% [markdown]
# This example uses the eyes-closed recording from ``ds004504`` used in the
# other EEGPhasePy tutorials. It selects simulated trigger samples at the
# positive peak, falling slope, trough, and rising slope of C3 alpha activity,
# then plots the average alpha-band voltage topography at each sample.
#
# The dataset contains resting EEG, not TMS trials. These are retrospective,
# zero-phase-filtered pseudo-triggers for visualization; they do not simulate
# a real-time phase estimator or a TMS response.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal as signal
from EEGPhasePy import viz


# %%
dataset_dir = Path(__file__).resolve().parent / 'ds004504'
recording_path = (
    dataset_dir / 'sub-001' / 'eeg' /
    'sub-001_task-eyesclosed_eeg.set'
)
if not recording_path.exists():
    import openneuro

    openneuro.download(
        dataset='ds004504',
        tag='1.0.8',
        target_dir=str(dataset_dir),
        include=['sub-001']
    )

raw = mne.io.read_raw_eeglab(
    str(recording_path),
    preload=True,
    verbose=False
)

sampling_rate = int(raw.info['sfreq'])
eeg_data = raw.get_data()
c3_index = raw.ch_names.index('C3')


# %% [markdown]
# Filter all channels to the alpha band. The analytic phase at C3 is used to
# choose the pseudo-trigger times. With this Hilbert phase convention, phase
# 0 is the positive peak, :math:`\\pi/2` is the falling slope, :math:`\\pi`
# is the trough, and :math:`-\\pi/2` is the rising slope.

# %%
alpha_sos = signal.butter(
    4, [8, 12], btype='bandpass', fs=sampling_rate, output='sos')
alpha_data = signal.sosfiltfilt(alpha_sos, eeg_data, axis=-1)
c3_phase = np.angle(signal.hilbert(alpha_data[c3_index]))

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
# Use the channel locations embedded in the EEGLAB file. Coordinates are
# centered and scaled for the plotting helper. EEG data from MNE are in volts,
# so convert to microvolts for readable colorbar values.
#
# Each map chooses its own symmetric color range. Use the colorbar values, not
# color saturation alone, when comparing voltage magnitudes across phases.

# %%
layout = mne.channels.make_eeg_layout(raw.info)
layout_positions = {
    channel_name: layout.pos[index, :2]
    for index, channel_name in enumerate(layout.names)
}
positions = np.array([layout_positions[name] for name in raw.ch_names])
positions -= positions.mean(axis=0)
positions /= np.max(np.linalg.norm(positions, axis=1))

alpha_data_uv = alpha_data * 1e6
topomap_figures = {}

for phase_name, trigger_samples in trigger_samples_by_phase.items():
    trigger_voltages = alpha_data_uv[:, trigger_samples].T
    figure = viz.plot_topomap(trigger_voltages, positions)
    figure.axes[0].set_title(
        f'C3 alpha {phase_name.lower()} '
        f'(n={len(trigger_samples)})'
    )
    figure.axes[-1].set_ylabel('Mean alpha voltage (uV)')
    topomap_figures[phase_name] = figure

plt.show()
