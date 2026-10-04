import numpy as np
import pytest

from EEGPhasePy.utils.check import (
    _check_filter_coefficients,
    _check_real_array,
    _check_real_number,
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
