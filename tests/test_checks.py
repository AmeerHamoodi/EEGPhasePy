import numpy as np
import pytest
import scipy.signal as signal

from EEGPhasePy.utils.check import (
    _check_filter_coefficients,
    _check_real_array,
    _check_real_number,
    _check_triggers,
    _filter_form,
)


@pytest.mark.parametrize('value', [1, 1.5, np.int64(2), np.float32(3.5)])
def test_check_real_number_accepts_real_scalars(value):
    assert _check_real_number(value, 'value') == float(value)


@pytest.mark.parametrize(
    'value,exception,message',
    [
        (True, TypeError, 'limit must be a real number'),
        ('1', TypeError, 'limit must be a real number'),
        (np.nan, ValueError, 'limit must be finite'),
        (np.inf, ValueError, 'limit must be finite'),
        (-1, ValueError, 'limit must be non-negative'),
    ],
)
def test_check_real_number_rejects_invalid_values(value, exception, message):
    with pytest.raises(exception, match=message):
        _check_real_number(value, 'limit', non_negative=True)


def test_check_real_array_returns_finite_real_array():
    result = _check_real_array([1, 2.5], 'data', contents='samples')

    np.testing.assert_array_equal(result, [1, 2.5])


@pytest.mark.parametrize(
    'value,exception,message',
    [
        ([1, [2]], ValueError, 'data must be a rectangular numeric array'),
        (['x'], TypeError, 'data must contain real numeric samples'),
        ([1 + 1j], TypeError, 'data must contain real numeric samples'),
        ([np.nan], ValueError, 'data must contain only finite samples'),
    ],
)
def test_check_real_array_rejects_invalid_values(value, exception, message):
    with pytest.raises(exception, match=message):
        _check_real_array(value, 'data', contents='samples')


def test_check_filter_coefficients_accepts_fir_and_iir():
    fir = np.array([0.25, 0.5, 0.25])
    fir_coefficients, numerator_length, denominator_length = \
        _check_filter_coefficients(fir)
    np.testing.assert_array_equal(fir_coefficients, fir)
    assert (numerator_length, denominator_length) == (3, 1)

    iir = np.array([[1.0, 0.0], [1.0, -0.5]])
    iir_coefficients, numerator_length, denominator_length = \
        _check_filter_coefficients(iir)
    np.testing.assert_array_equal(iir_coefficients, iir)
    assert (numerator_length, denominator_length) == (2, 2)


@pytest.mark.parametrize(
    'value,exception,message',
    [
        ([np.array([1.0])], ValueError,
         'real_time_filter must be a 1D numerator or a 2-row'),
        ([[], []], ValueError,
         'real_time_filter coefficients cannot be empty'),
        ([[np.inf, 1.0], [1.0, 0.0]], ValueError,
         'real_time_filter must contain only finite coefficients'),
        ([['x'], ['1']], TypeError,
         'real_time_filter must contain real numeric coefficients'),
    ],
)
def test_check_filter_coefficients_rejects_invalid_values(
        value, exception, message):
    with pytest.raises(exception, match=message):
        _check_filter_coefficients(value)


def test_check_filter_coefficients_accepts_sos():
    # two sections give a (2, 6) array, the same shape as a 5th order
    # (numerator, denominator) pair. Column 3 (a0) of every section is 1
    sos = signal.butter(2, [8, 12], btype='bandpass', fs=500, output='sos')
    assert sos.shape == (2, 6)
    assert _filter_form(sos) == 'sos'

    coefficients, numerator_length, denominator_length = \
        _check_filter_coefficients(sos)
    np.testing.assert_array_equal(coefficients, sos)
    assert (numerator_length, denominator_length) == (5, 5)

    b, a = signal.butter(5, 10, fs=500)
    assert _filter_form(np.array([b, a])) == 'ba'


def test_check_filter_coefficients_rejects_unstable_filters():
    with pytest.raises(ValueError, match=r'unstable.*\|z\| = 2'):
        _check_filter_coefficients([[1.0, 0.0], [1.0, -2.0]])

    unstable_sos = np.array([[1.0, 0.0, 0.0, 1.0, -2.0, 0.0]])
    with pytest.raises(ValueError, match='unstable'):
        _check_filter_coefficients(unstable_sos)


@pytest.mark.parametrize('value,n_samples,expected', [
    ([0, 5, 9], 10, [0, 5, 9]),
    (np.array([1.0, 2.0]), 10, [1, 2]),
    ([], 10, []),
])
def test_check_triggers_accepts_valid_indices(value, n_samples, expected):
    triggers = _check_triggers(value, n_samples)
    assert triggers.dtype == np.int64
    np.testing.assert_array_equal(triggers, expected)


@pytest.mark.parametrize('value,exception', [
    ([-1], ValueError),
    ([10], ValueError),
    ([0.5], TypeError),
    ([np.nan], TypeError),
    ([True], TypeError),
])
def test_check_triggers_rejects_invalid_indices(value, exception):
    with pytest.raises(exception):
        _check_triggers(value, 10)
