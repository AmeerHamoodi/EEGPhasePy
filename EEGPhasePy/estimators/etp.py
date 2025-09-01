import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from typing import Self
from .estimator import Estimator

from ..utils.check import _check_array_dimensions, _check_type


class ETP(Estimator):
    '''
    The Educated Temporal Prediction (ETP) model :cite:t:`Shirinpour2020-ef`

    ETP was first described by :cite:t:`Shirinpour2020-ef`. A more in depth
    description can be found there.

    Briefly, this EEG phase estimation model works by estimating the average inter-peak
    interval for the target EEG band. In real-time, ETP predicts the next time 
    the target phase will occur at (:math:`T_{adj}`)

    '''

    def __init__(self,
                 real_time_filter: np.ndarray,
                 ground_truth_filter: np.ndarray,
                 sampling_rate: int,
                 window_len=500,
                 window_edge=40):
        '''
        Construct a model for the educated-temporal-prediction (ETP) model of phase estimation (Shirinpour et al., 2020)

        Parameters
        ----------
        real_time_filter : array_like shape (n_parameters) | array_like shape (2, n_parameters)\n
            Filter parameters for filter to apply for predicting phase (should be constructed with fs of 1000).
            Accounts for FIR or IIR filters\n
        ground_truth_filter : array_like shape (n_parameters) | array_like shape (2, n_parameters)\n
            Filter parameters for filter to use during ETP training (should be constructed with fs of 1000).
            Accounts for FIR or IIR filters\n
        sampling_rate : int
            Original sampling rate of data. As per Shirinpour et al., 2020, data are downsampled to 1kHz\n
        window_len : 500 | int
            Window length in ms. Optional parameter to specify window length to train ETP with. This should match whatever is used in real-time\n
        window_edge : 40 | int
            Window edge to remove in ms. Optional parameter to specify edge to remove after applying real_time_filter\n
        '''
        super().__init__(real_time_filter, ground_truth_filter,
                         sampling_rate, window_len, window_edge)
        self.Tadj: int

    def fit(self, training_data: np.ndarray | list, min_ipi: int) -> Self:
        '''
        Estimates ideal Tadj for training data and updates self object with new Tadj.

        In cases of high phase instability, ETP may not converge onto the ideal Tadj. Using
        a more aggressive ground-truth filter typically helps

        Parameters
        ----------
        training_data : array (n_samples,)
            1D samples array of channel to estimate Tadj for
        min_ipi : int
            Minimum inter peak interval, should be the period of the upper frequency of the target band


        Returns
        -------
        self
        '''
        _check_type(training_data, ["array"])
        _check_type(min_ipi, ["int"])
        # ensure training data is 1D
        _check_array_dimensions(training_data, [(1,)])

        fs = self.sampling_rate
        # resampled_training_data = signal.resample(training_data, int(fs*len(training_data) / self.sampling_rate))
        filtered_data = self._filter_data(
            self.ground_truth_filter, training_data)
        # ground truth is hard to define for phase estimation, see Zrenner et al., 2020 for a more detailed discussion
        ground_truth_phase = np.angle(
            signal.hilbert(filtered_data), deg=True) % 360

        training_window_data = filtered_data[:90*fs]
        peaks = signal.find_peaks(training_window_data)[0]
        inter_peak_interval = np.diff(peaks)
        # remove IPIs that are too short
        inter_peak_interval = inter_peak_interval[inter_peak_interval > min_ipi]
        period = round(np.exp(np.nanmean(np.log(inter_peak_interval))))

        bias = 0
        bias_direction = None
        last_mean = None
        mean_differences = []

        window_len = int((self.window_len / 1000) * fs)

        while True:
            triggered_phases = []

            for i in range(255):
                window_i = 90*fs + 350*i
                window_data = training_data[window_i:window_i + window_len]
                filtered_window = self._filter_data(self.real_time_filter, window_data)[
                    :-self.window_edge]

                peaks = signal.find_peaks(filtered_window)[0]
                trigger_i = window_i + peaks[-1] + period + bias
                triggered_phases.append(ground_truth_phase[trigger_i])

            mean_phase = stats.circmean(np.deg2rad(triggered_phases))

            if bias_direction == None:
                bias_direction = 1 if np.rad2deg(
                    mean_phase) % 360 > 180 else -1

            if not last_mean == None:
                difference_mean = 1 - \
                    np.real(np.exp(1j*last_mean - 1j*mean_phase))
                if len(mean_differences) > 0 and mean_differences[-1] < difference_mean:
                    self.Tadj = period + bias - bias_direction
                    return self

                mean_differences.append(difference_mean)
            last_mean = mean_phase

            bias += bias_direction

    def predict(self, data: np.ndarray | list, target_phase: float) -> np.int64:
        '''
        Predicts the next sample target phase occurs at

        Parameters
        ----------
        data : array_like (n_samples)
            Window of EEG data from target channel to predict from
        target_phase : float
            Target phase to predict in radians


        Returns
        -------
        relative_next_phase : int
            Next sample target phase occurs, defined relative to window start
        '''

        _check_type(data, ["array"])
        _check_type(target_phase, ["float", "int"])

        _check_array_dimensions(data, [(1,)])

        filtered_window = self._filter_data(self.real_time_filter, data)
        peaks = signal.find_peaks(filtered_window)[0]

        if len(peaks) == 0:
            raise RuntimeError(
                "No peaks could be found in the window passed into the `predict` method")

        Tadj: int = int(self.Tadj * target_phase/2*np.pi)

        return peaks[-1] + Tadj
