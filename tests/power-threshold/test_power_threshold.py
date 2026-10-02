import numpy as np
import pytest
import scipy.signal as signal

from EEGPhasePy.estimators import Estimator


@pytest.fixture(params=['fir', 'iir'])
def estimator_and_coefficients(request):
    sampling_rate = 500
    if request.param == 'fir':
        numerator = signal.firwin(
            100, [8, 13], fs=sampling_rate, pass_zero=False)
        denominator = 1.0
        real_time_filter = numerator
    else:
        numerator, denominator = signal.butter(
            3, [8, 13], btype='bandpass', fs=sampling_rate)
        real_time_filter = np.array([numerator, denominator])

    estimator = Estimator(numerator, numerator, sampling_rate)
    estimator.real_time_filter = real_time_filter
    return estimator, numerator, denominator


@pytest.mark.parametrize('frequency,amplitude', [(10, 1), (10, 2), (30, 1)])
def test_power_threshold_matches_bandpass_hilbert_pipeline(
        estimator_and_coefficients, frequency, amplitude):
    estimator, numerator, denominator = estimator_and_coefficients
    time = np.arange(0, 2, 1 / estimator.sampling_rate)
    window = amplitude * np.sin(2 * np.pi * frequency * time)

    filtered = signal.filtfilt(numerator, denominator, window)
    edge = int(estimator.window_edge * estimator.sampling_rate / 1000)
    envelope = np.abs(signal.hilbert(filtered))[edge:-edge]
    expected_power = np.mean(envelope ** 2)

    assert estimator.check_power_threshold(window, expected_power) is True
    assert estimator.check_power_threshold(
        window, np.nextafter(expected_power, np.inf)) is False

    if frequency == 10:
        assert expected_power == pytest.approx(amplitude ** 2, rel=0.1)
    else:
        assert expected_power < 0.01


def test_power_threshold_scales_with_amplitude_squared(
        estimator_and_coefficients):
    estimator, _, _ = estimator_and_coefficients
    time = np.arange(0, 2, 1 / estimator.sampling_rate)
    base_window = np.sin(2 * np.pi * 10 * time)
    edge = int(estimator.window_edge * estimator.sampling_rate / 1000)

    def estimate_power(window):
        filtered = estimator._filter_data(estimator.real_time_filter, window)
        envelope = np.abs(signal.hilbert(filtered))[edge:-edge]
        return np.mean(envelope ** 2)

    assert estimate_power(2 * base_window) == pytest.approx(
        4 * estimate_power(base_window))


def test_zero_signal_and_zero_threshold_pass():
    numerator, denominator = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([numerator, denominator])

    assert estimator.check_power_threshold(np.zeros(100), 0) is True
    assert estimator.check_power_threshold(np.zeros(100), 1e-12) is False


@pytest.mark.parametrize(
    ('filter_type', 'length', 'expected_minimum'),
    [('fir', 300, 301), ('iir', 40, 41)],
)
def test_power_threshold_rejects_windows_too_short_for_filtering_or_edges(
        filter_type, length, expected_minimum):
    if filter_type == 'fir':
        numerator = signal.firwin(100, [8, 13], fs=500, pass_zero=False)
    else:
        numerator, denominator = signal.butter(
            1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    if filter_type == 'iir':
        estimator.real_time_filter = np.array([numerator, denominator])

    with pytest.raises(
            ValueError,
            match=rf'data window is too short: expected at least {expected_minimum} samples'):
        estimator.check_power_threshold(np.zeros(length), 0)

    assert isinstance(
        estimator.check_power_threshold(np.zeros(expected_minimum), 0), bool)


@pytest.mark.parametrize(
    'window,exception,message',
    [
        (np.zeros((2, 100)), ValueError, 'wrong dimension.*1D'),
        (np.array(['bad'] * 100), TypeError, 'real numeric samples'),
        (np.array([1 + 1j] * 100), TypeError, 'real numeric samples'),
        (np.array([np.nan] * 100), ValueError, 'finite samples'),
        (np.array([np.inf] * 100), ValueError, 'finite samples'),
        ([1, 2, [3]], ValueError, 'rectangular numeric array'),
    ],
)
def test_power_threshold_rejects_invalid_windows(window, exception, message):
    numerator, denominator = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([numerator, denominator])

    with pytest.raises(exception, match=message):
        estimator.check_power_threshold(window, 0)


@pytest.mark.parametrize(
    'threshold,exception,message',
    [
        ('0.4', TypeError, 'power_threshold must be a real number'),
        (True, TypeError, 'power_threshold must be a real number'),
        (np.nan, ValueError, 'power_threshold must be finite'),
        (np.inf, ValueError, 'power_threshold must be finite'),
        (-0.1, ValueError, 'power_threshold must be non-negative'),
    ],
)
def test_power_threshold_rejects_invalid_thresholds(
        threshold, exception, message):
    numerator, denominator = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([numerator, denominator])

    with pytest.raises(exception, match=message):
        estimator.check_power_threshold(np.zeros(100), threshold)


def test_power_threshold_accepts_numpy_real_scalar_threshold():
    numerator, denominator = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([numerator, denominator])

    assert estimator.check_power_threshold(
        np.zeros(100), np.float32(0)) is True


def test_power_threshold_rejects_non_finite_filter_coefficients():
    numerator, denominator = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([[np.nan, 0.0], [1.0, 0.0]])

    with pytest.raises(
            ValueError,
            match='real_time_filter coefficients must be finite'):
        estimator.check_power_threshold(np.zeros(100), 0)


def test_power_threshold_rejects_malformed_filter_shape():
    numerator, _ = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([numerator])

    with pytest.raises(
            ValueError,
            match='real_time_filter must be a 1D numerator or a 2-row'):
        estimator.check_power_threshold(np.zeros(100), 0)


def test_power_threshold_reports_overflow_in_computed_power():
    numerator, denominator = signal.butter(
        1, [8, 13], btype='bandpass', fs=500)
    estimator = Estimator(numerator, numerator, 500)
    estimator.real_time_filter = np.array([numerator, denominator])
    time = np.arange(100) / estimator.sampling_rate
    window = 1e200 * np.sin(2 * np.pi * 10 * time)

    with pytest.raises(
            ValueError,
            match='computed power is non-finite; check the signal scale'):
        estimator.check_power_threshold(window, 0)