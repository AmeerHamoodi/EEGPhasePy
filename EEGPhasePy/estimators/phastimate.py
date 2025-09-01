import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from typing import Self
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import AutoReg

from .estimator import Estimator
from ..utils.check import _check_array_dimensions, _check_type


class PHASTIMATE(Estimator):
    '''
    Class for the PHASTIMATE algorithm created by :cite:t:`Zrenner2020-zb`. If you use this class
    please cite :cite:t:`Zrenner2020-zb`

    PHASTIMATE uses an autoregressive approach to fill in data impacted by filter edge effects
    then applys a hilbert transform to extract the phase at the current time. It is best used 
    in real-time environments where minimal delay in your system (particularly 
    EEG -> computer) is guarenteed, because forward forecasting more than a couple of milliseconds
    beyond t = 0, typically results in inaccurate phase predictions. This implementation does not
    support forward forecasting beyond t = 0

    PHASTIMATE also comes with genetic optimization for the phase estimation algorithms including
    order, window_edge, window_len. Our implementation of PHASTIMATE does not include optimization
    over filter parameters.
    '''

    def __init__(self, real_time_filter: np.ndarray,
                 ground_truth_filter: np.ndarray,
                 sampling_rate: int,
                 window_len=500,
                 window_edge=40,
                 ar_order=10):
        '''
        Constructor for PHASTIMATE class

        Parameters
        -----------
        real_time_filter : array_like shape (n_parameters) | array_like shape (2, n_parameters)\n
            Filter parameters for filter to apply for predicting phase (should be constructed with fs of 1000).
            Accounts for FIR or IIR filters\n
        ground_truth_filter : array_like shape (n_parameters) | array_like shape (2, n_parameters)\n
            Filter parameters for filter to use during ETP training (should be constructed with fs of 1000).
            Accounts for FIR or IIR filters\n
        sampling_rate : int
            Original sampling rate of data.
        window_len : 500 | int
            Window length in ms. Optional parameter to specify window length to train ETP with (not used in this estimator). 
            This should match whatever is used in real-time\n
        window_edge : 40 | int
            Window edge to remove in ms. Optional parameter to specify edge to remove after applying real_time_filter\n
        ar_order : 10 | int
            The order for the auto-regressive model
        '''
        super().__init__(real_time_filter, ground_truth_filter,
                         sampling_rate, window_len, window_edge)

        _check_type(ar_order, ['int'])
        self.ar_order = ar_order

    def _ar_forecast(self, data, ar_params, steps=10) -> np.ndarray:
        """
        Forecast future values from an AR process.

        Parameters
        ----------
        data : array-like
            Time series data.
        ar_params : array-like
            AR coefficients (phi_1, ..., phi_p).
        steps : int
            Number of steps to forecast.

        Returns
        --------
        forecasted_data : ndarray
            The forecasted series
        """
        p = len(ar_params)
        forecast = list(data[-p:])  # start with the last p values

        for _ in range(steps):
            new_val = np.dot(ar_params, forecast[-p:][::-1])  # weighted sum
            forecast.append(new_val)

        return np.array(forecast[p:])

    def predict(self, data: np.ndarray[float] | list[float], target_phase: float | int, tolerance: float | int = 5) -> bool:
        '''
        Predict whether the phase at the current time matches the target phase

        Parameters
        -----------
        data : np.ndarray[float] | list[float] 
            The (n_samples,) array containing the unfiltered EEG data in the current window
        target_phase : float | int
            The phase in degrees that the current phase should match
        tolerance : float | int
            The tolerance between the current phase and target phase. The target phase has to be
            within `tolerance` degrees for this method to return `True`

        Returns
        --------
        current_phase_matches_target : bool
            Whether the current phase matches the target phase with a given tolerance

        '''
        _check_type(data, ['array'])
        _check_type(target_phase, ['int', 'float'])
        _check_type(tolerance, ['int', 'float'])
        _check_array_dimensions(data, [(1,)])

        edge = int((self.window_edge / 1000) * self.sampling_rate)
        filtered_data = self._filter_data(
            self.real_time_filter, data)[edge:-edge]

        ar_params, _ = yule_walker(filtered_data, self.ar_order)
        forecasted_data = self._ar_forecast(filtered_data, ar_params, 2*edge)
        full_data = np.concatenate([filtered_data, forecasted_data])

        analytic_signal = signal.hilbert(full_data)
        phase_t0 = np.angle(analytic_signal[-edge + 1], deg=True) % 360

        return np.isclose(phase_t0, target_phase % 360, atol=tolerance) or np.isclose(phase_t0, 360 - (target_phase % 360), atol=tolerance)
