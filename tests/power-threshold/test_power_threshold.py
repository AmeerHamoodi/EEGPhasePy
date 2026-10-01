import numpy as np
import pytest
import scipy.signal as signal

from EEGPhasePy.estimators import Estimator


def test_estimator_check_power_threshold():
    custom_filter = signal.firwin(100, [8, 13], fs=500, pass_zero=False)
    estimator = Estimator(custom_filter, custom_filter, 500)
    time = np.arange(0, 2, 1 / 500)
    window = np.sin(2 * np.pi * 10 * time)

    assert estimator.check_power_threshold(window, 0.4) is True
    assert estimator.check_power_threshold(window, 0.6) is False

    with pytest.raises(TypeError, match='Value must be one of: int or float'):
        estimator.check_power_threshold(window, '0.4')
    with pytest.raises(ValueError, match='power_threshold must be non-negative'):
        estimator.check_power_threshold(window, -0.1)