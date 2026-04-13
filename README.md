# EEGPhasePy

[![Tests](https://github.com/AmeerHamoodi/EEGPhasePy/actions/workflows/lint-and-test.yml/badge.svg)](https://github.com/AmeerHamoodi/EEGPhasePy/actions/workflows/lint-and-test.yml)
[![codecov](https://codecov.io/gh/AmeerHamoodi/EEGPhasePy/branch/master/graph/badge.svg)](https://codecov.io/gh/AmeerHamoodi/EEGPhasePy)
[![PyPI version](https://img.shields.io/pypi/v/eegphasepy)](https://pypi.org/project/eegphasepy/)
[![Python versions](https://img.shields.io/pypi/pyversions/eegphasepy)](https://pypi.org/project/eegphasepy/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A toolkit for developing and analyzing real-time and pseudo-real-time EEG phase estimation.

EEGPhasePy includes implementations of various EEG phase estimation algorithms — including Educated Temporal Prediction (ETP) and PHASTIMATE (autoregressive) — alongside offline analysis helpers for evaluating phase estimation experiments. For background on EEG phase estimation and its applications, see [Zrenner & Ziemann (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10881194/). More phase estimators will continue to be added in the future.

## Features

- **ETP** — Educated Temporal Prediction ([Shirinpour et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC8293904/)): uses average interpeak-interval to determine time to next target phase. Similar accuracy to AR with added bonus of being useful for amplifiers that have low packet send rates.
- **PHASTIMATE** — Autoregressive phase estimator ([Zrenner et al., 2020](https://pubmed.ncbi.nlm.nih.gov/29191438/)): uses an AR model to compensate for filter edge effects and extracts instantaneous phase via the Hilbert transform. Estimates ongoing phase rather than predicting.
- **Parameter optimization** — Bayesian and genetic optimization (via `bayesian-optimization` and `PyGAD`) over AR order and window edge for PHASTIMATE. Improve accuracy of AR model on a participant-by-participant basis.
- **Visualization** — Polar phase histograms and average ± std pre/post-trigger waveforms via `matplotlib`.
- Supports Python 3.9–3.13.

## Installation

```bash
pip install eegphasepy
```

## Quick Start

### ETP

```python
import numpy as np
import scipy.signal as signal
from EEGPhasePy.estimators import ETP

fs = 5000  # Hz
# Build bandpass filter for the target frequency band (e.g. alpha, 8-12 Hz)
rt_filter = signal.firwin(101, [8, 12], pass_zero=False, fs=fs)
gt_filter = signal.firwin(201, [8, 12], pass_zero=False, fs=fs)

estimator = ETP(
    real_time_filter=rt_filter,
    ground_truth_filter=gt_filter,
    sampling_rate=fs,
    window_len=500,   # ms
    window_edge=40,   # ms
)

# Fit on training data (1D array, at least ~180 s at fs)
estimator.fit(training_data, min_ipi=int(fs / 12))

# Predict next sample at which the target phase (e.g. peak, 0 rad) will occur
next_sample = estimator.predict(current_window, target_phase=0.0)
```

### PHASTIMATE

```python
from EEGPhasePy.estimators import PHASTIMATE

estimator = PHASTIMATE(
    real_time_filter=rt_filter,
    ground_truth_filter=gt_filter,
    sampling_rate=fs,
    window_len=500,
    window_edge=40,
    ar_order=30,
)

# Optionally optimize window_edge and ar_order on training data
estimator.optimize_parameters(training_data, method="bayesian")  # or "genetic"

# At each time step: returns True if current phase matches target within tolerance
is_target = estimator.predict(current_window, target_phase=0, tolerance=5)
```

### Visualization

```python
from EEGPhasePy.viz import plot_polar_histogram, plot_waveform_average

# Polar histogram of triggered phases (radians)
fig = plot_polar_histogram(triggered_phases)
fig.show()

# Average ± std waveform around trigger events
fig = plot_waveform_average(waveform_segments, fs=fs, t_trigger=trigger_sample)
fig.show()
```

## Documentation

Full documentation including API reference, usage guides, and examples is available at [eegphasepy.readthedocs.io](https://eegphasepy.readthedocs.io).

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/AmeerHamoodi/EEGPhasePy).

## Citation

If you use EEGPhasePy in your research, please cite the underlying algorithm(s) you end up using:

- **ETP**: Shirinpour et al. (2020). _Experimental Evaluation of Methods for Real-Time EEG Phase-Specific Transcranial Magnetic Stimulation_. [PMC8293904](https://pmc.ncbi.nlm.nih.gov/articles/PMC8293904/)
- **PHASTIMATE**: Zrenner et al. (2020). _The shaky ground truth of real-time phase estimation_. [PMID 29191438](https://pubmed.ncbi.nlm.nih.gov/29191438/)

## License

BSD 3-Clause License. See [LICENSE](LICENCSE) for details.
