from EEGPhasePy.estimators import PHASTIMATE
import pytest
import numpy as np
import scipy.signal as signal

from EEGPhasePy.utils import check


def test_phastimate_construct():
    rt_filter = signal.firwin(200, [8, 12], pass_zero=False, fs=500)
    gt_filter = signal.firwin(300, [8, 12], pass_zero=False, fs=500)

    phastimate_paramset = [
        [rt_filter, gt_filter, 500, 500, 40, 1.5],
        [rt_filter, gt_filter, 500, 500, 40, 35]
    ]

    phastimate_output = [
        [TypeError, 'Value must be an int type'],
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

    phastimate = PHASTIMATE(rt_filter, gt_filter, fs,
                            ar_order=12, window_edge=64)

    time_data = np.arange(0, 2, 1/fs)

    clean_signal = np.sin(2 * np.pi * 10 * time_data)
    ground_truth_phase = np.angle(signal.hilbert(clean_signal), deg=True) % 360

    # local seeded generator so the result does not depend on test ordering
    # (other test modules reseed the global numpy RNG on import)
    rng = np.random.default_rng(123)
    minimally_noisy_alpha = clean_signal + \
        rng.normal(0, 0.1, size=len(time_data))

    phastimate_predict_paramset = [
        [100, 25],
        [minimally_noisy_alpha, [25]],
        [[minimally_noisy_alpha], 25],
        [minimally_noisy_alpha, 25, [100]],
        [minimally_noisy_alpha[-500:-250], ground_truth_phase[-220], 10],
        [minimally_noisy_alpha[-500:-250], ground_truth_phase[-250], 10],
        [minimally_noisy_alpha[-470:-220], ground_truth_phase[-220], 10],
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


def test_phastimate_ar_fit_is_stable():
    fs = 500
    rt_filter = signal.firwin(80, [8, 12], pass_zero=False, fs=fs)
    gt_filter = signal.firwin(300, [8, 12], pass_zero=False, fs=fs)

    phastimate = PHASTIMATE(rt_filter, gt_filter, fs, ar_order=30)

    edge = int((phastimate.window_edge / 1000) * fs)
    window_len = int((phastimate.window_len / 1000) * fs)
    time_data = np.arange(window_len) / fs
    rng = np.random.default_rng(0)

    for _ in range(100):
        window = np.sin(2 * np.pi * 10 * time_data + rng.uniform(0, 2 * np.pi)) + \
            rng.normal(0, 2, size=window_len)
        filtered_data = phastimate._filter_data(rt_filter, window)[edge:-edge]

        ar_params = phastimate._fit_ar(filtered_data)
        poles = np.roots(np.concatenate([[1], -ar_params]))

        assert np.all(np.abs(poles) < 1)


@pytest.mark.parametrize("phase, target_phase, expected", [
    (354, 2, False),
    (1, 358, True),
    (359, 2, True),
    (3, -2, True),
    (180, 0, False),
    (90, 92, True),
    (90, 96, False),
    (2, 362, True),
])
def test_phastimate_predict_wraps_around(monkeypatch, phase, target_phase, expected):
    fs = 500
    rt_filter = signal.firwin(80, [8, 12], pass_zero=False, fs=fs)
    gt_filter = signal.firwin(300, [8, 12], pass_zero=False, fs=fs)

    phastimate = PHASTIMATE(rt_filter, gt_filter, fs)

    # pin the estimated phase so only the tolerance comparison is under test
    monkeypatch.setattr(signal, "hilbert",
                        lambda x: np.full(len(x), np.exp(1j * np.deg2rad(phase))))

    assert phastimate.predict(np.ones(250), target_phase, 5) == expected


def test_phastimate_estimate_phase():
    fs = 500
    rt_filter = signal.firwin(80, [8, 12], pass_zero=False, fs=fs)
    gt_filter = signal.firwin(300, [8, 12], pass_zero=False, fs=fs)

    phastimate = PHASTIMATE(rt_filter, gt_filter, fs,
                            ar_order=12, window_edge=64)

    time_data = np.arange(0, 2, 1/fs)
    clean_signal = np.sin(2 * np.pi * 10 * time_data)
    ground_truth_phase = np.angle(signal.hilbert(clean_signal), deg=True) % 360

    rng = np.random.default_rng(123)
    noisy_alpha = clean_signal + rng.normal(0, 0.1, size=len(time_data))

    with pytest.raises(TypeError, match=r"Value must be an array type"):
        phastimate.estimate_phase(100)
    with pytest.raises(ValueError, match=r"wrong dimension"):
        phastimate.estimate_phase([noisy_alpha])

    for end in [750, 780, 800]:
        window = noisy_alpha[end - 250:end]
        phase = phastimate.estimate_phase(window)

        assert isinstance(phase, float)
        assert 0 <= phase < 360
        # phase refers to the sample immediately after the window
        assert abs((phase - ground_truth_phase[end] + 180) % 360 - 180) < 15

        prediction = phastimate.predict(window, phase, 0.5)
        assert type(prediction) is bool
        assert prediction
