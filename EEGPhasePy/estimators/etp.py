import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import scipy.stats as stats
from typing import Union
from .estimator import Estimator

from ..utils.check import (
    _check_array_dimensions,
    _check_real_number,
    _check_type,
)


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
        self.period: int
        self.period = None

    def fit(self, training_data: npt.ArrayLike, min_ipi: int,
            ipi_duration: Union[int, float] = 90, n_windows: int = 255):
        '''
        Estimates ideal tadj for training data and updates self object with
        new tadj.

        The first ``ipi_duration`` seconds of ``training_data`` are used to
        estimate the average inter-peak interval (the period). ``n_windows``
        pseudo-real-time windows, spaced 350 ms apart, are then taken from
        the data that follows to calibrate tadj, the time from the last peak
        in a window to the next peak. ``training_data`` must be long enough
        for both, e.g. at least ~180 s with the defaults and a 500 ms window

        In cases of high phase instability, ETP may not converge onto the
        ideal tadj. Using a more aggressive ground-truth filter typically helps

        Parameters
        ----------
        training_data : array (n_samples,)
            1D samples array of channel to estimate tadj for
        min_ipi : int
            Minimum inter peak interval in number of samples, should be the
            period of the upper frequency of the target band. Shorter
            intervals are ignored when estimating the period
        ipi_duration : int | float
            Length in seconds of the start of ``training_data`` used to
            estimate the inter-peak interval
        n_windows : int
            Number of windows used to calibrate tadj

        Returns
        -------
        self
        '''
        _check_type(training_data, ["array"])
        _check_type(min_ipi, ["int"])
        _check_type(n_windows, ["int"])
        # ensure training data is 1D
        _check_array_dimensions(training_data, [(1,)])

        ipi_duration = _check_real_number(ipi_duration, 'ipi_duration',
                                          non_negative=True)
        if ipi_duration == 0:
            raise ValueError('ipi_duration must be positive')
        if n_windows < 1:
            raise ValueError('n_windows must be at least 1')

        fs = self.sampling_rate
        ipi_samples = int(ipi_duration * fs)
        window_len = int((self.window_len / 1000) * fs)
        window_edge = int((self.window_edge / 1000) * fs)
        window_step = int(0.35 * fs)

        required_samples = ipi_samples + \
            (n_windows - 1) * window_step + window_len
        if len(training_data) < required_samples:
            raise ValueError(
                f"training_data is too short: fitting with ipi_duration="
                f"{ipi_duration:g} s and n_windows={n_windows} needs at least "
                f"{required_samples} samples ({required_samples / fs:.1f} s), "
                f"got {len(training_data)} ({len(training_data) / fs:.1f} s). "
                "Pass more data or lower ipi_duration / n_windows")

        filtered_data = self._filter_data(
            self.ground_truth_filter, training_data)
        # ground truth is hard to define for phase estimation, see Zrenner
        # et al., 2020 for a more detailed discussion
        ground_truth_phase = np.angle(
            signal.hilbert(filtered_data), deg=True) % 360

        peaks = signal.find_peaks(filtered_data[:ipi_samples])[0]
        inter_peak_interval = np.diff(peaks)
        # remove IPIs that are too short
        inter_peak_interval = inter_peak_interval[inter_peak_interval >=
                                                  min_ipi]

        if len(inter_peak_interval) == 0:
            raise ValueError(
                f"No inter-peak intervals of at least min_ipi={min_ipi} "
                f"samples were found in the first {ipi_duration:g} s of "
                "training_data. Check that min_ipi is in samples and that "
                "ground_truth_filter passes the target band")

        period = round(np.exp(np.mean(np.log(inter_peak_interval))))

        # the last peak in each window only depends on the data, so find it
        # once and reuse it for every candidate tadj
        last_peaks = []
        for i in range(n_windows):
            window_i = ipi_samples + window_step * i
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

            last_peaks.append(window_i + peaks[-1])

        last_peaks = np.array(last_peaks)

        # search one full cycle of candidates around the period and keep the
        # tadj whose triggers land closest to the peak (0 degrees). Checking
        # every candidate means noise cannot stop the search early
        best_distance = np.inf
        for candidate in range(period - period // 2,
                               period + period // 2 + 1):
            trigger_i = last_peaks + candidate
            trigger_i = trigger_i[trigger_i < len(ground_truth_phase)]

            if len(trigger_i) == 0:
                continue

            mean_phase = stats.circmean(
                np.deg2rad(ground_truth_phase[trigger_i]))
            distance_to_target = 1 - np.cos(mean_phase)

            if distance_to_target < best_distance:
                best_distance = distance_to_target
                self.tadj = candidate

        self.period = period

        return self

    def predict(self, data: npt.ArrayLike, target_phase: Union[int, float],
                min_samples_to_wait: int = 0) -> np.int64:
        '''
        Predicts the next sample target phase occurs at

        Parameters
        ----------
        data : array_like (n_samples)
            Window of EEG data from target channel to predict from
        target_phase : int | float
            Target phase to predict in degrees. 0 is a peak and 180 is a
            trough. Any value is accepted and wrapped to [0, 360)
        min_samples_to_wait : int
            The earliest acceptable trigger, in samples after the window.
            Set this to your system's stimulation latency so the returned
            trigger can still be delivered on time

        Returns
        -------
        samples_to_wait : int
            Number of samples to wait for target phase to occur. Always at
            least ``min_samples_to_wait``
        '''
        if self.tadj is None:
            raise RuntimeError(
                "ETP has not been fit. Call `fit` with resting-state data " +
                "before predicting")

        _check_type(data, ["array"])
        _check_type(target_phase, ["float", "int"])
        _check_type(min_samples_to_wait, ["int"])

        _check_array_dimensions(data, [(1,)])

        if min_samples_to_wait < 0:
            raise ValueError("min_samples_to_wait must be non-negative")

        window_edge = int((self.window_edge / 1000) * self.sampling_rate)

        filtered_window = self._filter_data(self.real_time_filter, data)
        peaks = signal.find_peaks(filtered_window[:-window_edge])[0]

        if len(peaks) == 0:
            raise RuntimeError(
                "No peaks could be found in the window passed into the "
                "`predict` method")

        # measure the target from the fitted next-peak time so the bias
        # correction in tadj applies to every target phase, not just 0
        trigger_offset = self.tadj + \
            round(self.period * (target_phase % 360) / 360)
        samples_to_wait = trigger_offset - (len(filtered_window) - peaks[-1])

        # step by whole periods to the next occurrence that is no sooner than
        # min_samples_to_wait, so the trigger is never in the past
        samples_to_wait = min_samples_to_wait + \
            (samples_to_wait - min_samples_to_wait) % self.period

        return np.int64(samples_to_wait)
