from EEGPhasePy.estimators import PHASTIMATE
import pytest
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import mne

from EEGPhasePy.utils import check


def test_phastimate_construct():
    rt_filter = signal.firwin(200, [8, 12], pass_zero=False, fs=500)
    gt_filter = signal.firwin(300, [8, 12], pass_zero=False, fs=500)

    phastimate_paramset = [
        [rt_filter, gt_filter, 500, 500, 40, 1.5],
        [rt_filter, gt_filter, 500, 500, 40, 35]
    ]

    phastimate_output = [
        [TypeError, 'Value must be an int'],
        False
    ]

    for i, paramset in enumerate(phastimate_paramset):
        if check._is_array(phastimate_output[i]):
            with pytest.raises(phastimate_output[i][0], match=r"" + phastimate_output[i][1] + ""):
                phastimate = PHASTIMATE(*paramset)
        else:
            phastimate = PHASTIMATE(*paramset)


def test_phastimate_predict():
    fs = 500
    rt_filter = signal.firwin(80, [8, 12], pass_zero=False, fs=fs)
    gt_filter = signal.firwin(300, [8, 12], pass_zero=False, fs=fs)

    phastimate = PHASTIMATE(rt_filter, gt_filter, fs)

    time_data = np.arange(0, 2, 1/fs)

    clean_signal = np.sin(2 * np.pi * 10 * time_data)
    ground_truth_phase = np.angle(signal.hilbert(clean_signal), deg=True) % 360

    minimally_noisy_alpha = clean_signal + \
        np.random.normal(0, 1, size=len(time_data))

    phastimate_predict_paramset = [
        [100, 25],
        [minimally_noisy_alpha, [25]],
        [[minimally_noisy_alpha], 25],
        [minimally_noisy_alpha, 25, [100]],
        [minimally_noisy_alpha[-500:-250], ground_truth_phase[-200], 5],
        [minimally_noisy_alpha[-250:], ground_truth_phase[-250], 5],
        [minimally_noisy_alpha[-470:-220], ground_truth_phase[-220], 5],
        [minimally_noisy_alpha[-470:-220], ground_truth_phase[-200], 0.5],
    ]

    phastimate_output = [
        [TypeError, "Value must be an array type"],
        [TypeError, "Value must be one of: int or float"],
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [TypeError, "Value must be one of: int or float"],
        np.False_,
        np.True_,
        np.True_,
        np.False_
    ]

    for i, param_set in enumerate(phastimate_predict_paramset):
        if check._is_array(phastimate_output[i]) and (phastimate_output[i][0] == ValueError or phastimate_output[i][0] == TypeError):
            with pytest.raises(phastimate_output[i][0], match=r"" + phastimate_output[i][1] + ""):
                phase_match = phastimate.predict(*param_set)
        else:
            phase_match = phastimate.predict(*param_set)
            assert phase_match == phastimate_output[i]
