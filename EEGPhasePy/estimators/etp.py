import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import scipy.stats as stats
from typing import Union
from .estimator import Estimator

from ..utils.check import _check_array_dimensions, _check_type


class ETP(Estimator):
    '''
    The Educated Temporal Prediction (ETP) model :cite:t:`Shirinpour2020-ef`.
    If you use this class please cite :cite:t:`Shirinpour2020-ef`

    ETP was first described by :cite:t:`Shirinpour2020-ef`. A more in depth
    description can be found there.

    Briefly, this EEG phase estimation model works by estimating the average
    inter-peak interval for the target EEG band. In real-time, ETP predicts
    the next time the target phase will occur at (:math:`T_{adj}`)
    '''

    def __init__(self,
                 real_time_filter: np.ndarray,
                 ground_truth_filter: np.ndarray,
                 sampling_rate: int,
                 window_len: int = 500,
                 window_edge: int = 40):
        '''
        Construct a model for the educated-temporal-prediction (ETP) model of
        phase estimation :cite:t:`Shirinpour2020-ef`

        Parameters
        ----------
        real_time_filter
            Filter parameters for filter to apply for predicting phase.
            Accounts for FIR or IIR filters
        ground_truth_filter
            Filter parameters for identifying "true" phase. Used
            to fit ETP. See :cite:t:Zrenner2020-zb for an
            in-depth discussion on "true" phase. Accounts for FIR
            or IIR filters
        sampling_rate : int
            Original sampling rate of data.
        window_len : 500 | int
            Window length in ms. Optional parameter to specify window length
            to train ETP with. This should match whatever is used in real-time
        window_edge : 40 | int
            Window edge to remove in ms. Optional parameter to specify edge to
            remove after applying real_time_filter
        '''
        super().__init__(real_time_filter, ground_truth_filter,
                         sampling_rate, window_len, window_edge)
        self.tadj: int
        self.tadj = None

    def fit(self, training_data: npt.ArrayLike, min_ipi: int):
        '''
        Estimates ideal tadj for training data and updates self object with
        new tadj.

        In cases of high phase instability, ETP may not converge onto the
        ideal tadj. Using a more aggressive ground-truth filter typically helps

        Parameters
        ----------
        training_data : array (n_samples,)
            1D samples array of channel to estimate tadj for
        min_ipi : int
            Minimum inter peak interval in number of samples, should be the
            period of the upper frequency of the target band


        Returns
        -------
        self
        '''
        _check_type(training_data, ["array"])
        _check_type(min_ipi, ["int"])
        # ensure training data is 1D
        _check_array_dimensions(training_data, [(1,)])

        fs = self.sampling_rate
        filtered_data = self._filter_data(
            self.ground_truth_filter, training_data)
        # ground truth is hard to define for phase estimation, see Zrenner
        # et al., 2020 for a more detailed discussion
        ground_truth_phase = np.angle(
            signal.hilbert(filtered_data), deg=True) % 360

        training_window_data = filtered_data[:90*fs]
        peaks = signal.find_peaks(training_window_data)[0]
        inter_peak_interval = np.diff(peaks)
        # remove IPIs that are too short
        inter_peak_interval = inter_peak_interval[inter_peak_interval >
                                                  min_ipi]
        period = round(np.exp(np.nanmean(np.log(inter_peak_interval))))

        bias = 0
        bias_direction = None
        last_mean = None
        mean_differences = []

        window_len = int((self.window_len / 1000) * fs)
        window_edge = int((self.window_edge / 1000) * fs)

        n_fitting_iterations = 0

        while True:
            if n_fitting_iterations > 100:
                raise TimeoutError(
                    "ETP fitting exceeded maximum allowable iterations")

            triggered_phases = []

            window_step = int(0.35 * fs)
            for i in range(255):
                window_i = 90 * fs + window_step * i
                window_data = training_data[window_i:window_i + window_len]
                filtered_window = self._filter_data(
                    self.real_time_filter,
                    window_data)[:-window_edge]

                peaks = signal.find_peaks(filtered_window)[0]

                if len(peaks) == 0:
                    raise ValueError(
                        "No peaks were found during ETP fitting. This could be"
                        + " one of: a signal quality issue (try increasing"
                        + " filter order), window length issue (try increasing"
                        + " the window length)")

                trigger_i = window_i + peaks[-1] + period + bias

                if trigger_i < len(ground_truth_phase) - 1:
                    triggered_phases.append(ground_truth_phase[trigger_i])

            mean_phase = stats.circmean(np.deg2rad(triggered_phases))

            if bias_direction is None:
                bias_direction = 1 if np.rad2deg(
                    mean_phase) % 360 > 180 else -1

            if last_mean is not None:
                difference_mean = 1 - \
                    np.real(np.exp(1j*last_mean - 1j*mean_phase))
                if len(mean_differences) > 0 and \
                        mean_differences[-1] < difference_mean:
                    self.tadj = period + bias - bias_direction
                    return self

                mean_differences.append(difference_mean)
            last_mean = mean_phase

            bias += bias_direction
            n_fitting_iterations += 1

    def predict(self, data: npt.ArrayLike, target_phase: Union[int, float]) \
            -> np.int64:
        '''
        Predicts the next sample target phase occurs at

        Parameters
        ----------
        data : array_like (n_samples)
            Window of EEG data from target channel to predict from
        target_phase : int | float
            Target phase to predict in degrees


        Returns
        -------
        samples_to_wait : int
            Number of samples to wait for target phase to occur
        '''

        _check_type(data, ["array"])
        _check_type(target_phase, ["float", "int"])

        _target_phase = np.deg2rad(target_phase)

        _check_array_dimensions(data, [(1,)])

        window_edge = int((self.window_edge / 1000) * self.sampling_rate)

        filtered_window = self._filter_data(self.real_time_filter, data)
        peaks = signal.find_peaks(filtered_window[:-window_edge])[0]

        if len(peaks) == 0:
            raise RuntimeError(
                "No peaks could be found in the window passed into the \
                    `predict` method")

        tadj: int = int(self.tadj * _target_phase/(2*np.pi))

        return tadj - (len(filtered_window) - peaks[-1])
