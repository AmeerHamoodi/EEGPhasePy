import hashlib
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import scipy.stats as stats
import matplotlib
from typing import Optional, Union

from ..utils.check import (
    _check_array_dimensions,
    _check_filter_coefficients,
    _check_real_array,
    _check_real_number,
    _check_triggers,
    _check_type,
    _filter_form,
)
from ..viz import plot_polar_histogram, plot_waveform_average


class Estimator:
    '''
      Base model for all phase estimators. This class contains the helper
      functions, validation and base parameters needed for all estimators.

      All estimators share one timing convention: the "current time" is the
      sample immediately after the last sample of the window passed in.
      Phases from ``estimate_phase`` and ``predict`` refer to that sample,
      and ``ETP.predict`` counts ``samples_to_wait`` from it. When simulating
      offline on a recording, a window ``data[start:start + n]`` therefore
      corresponds to a trigger at sample ``start + n``.

      All estimators also share one phase convention: degrees in
      [0, 360), where 0 is a peak and 180 is a trough. Target phases passed
      to ``predict``, ``phase_accuracy_from_triggers`` and
      ``compute_phase_stats`` are in degrees, and every method that returns
      a phase returns degrees in [0, 360) by default. Methods with a
      ``degree`` argument return radians in [0, :math:`2\\pi`) when it is
      False.

      ``real_time_filter`` and ``ground_truth_filter`` can be given as a 1D
      FIR numerator, a 2-row (numerator, denominator) array, or an
      (n_sections, 6) array of second-order sections (SOS), as returned by
      ``scipy.signal`` filter design functions with ``output='sos'``. SOS is
      recommended for IIR filters: high-order IIR filters in (numerator,
      denominator) form are numerically unstable and are rejected.
    '''

    def __init__(self,
                 real_time_filter: npt.ArrayLike,
                 ground_truth_filter: npt.ArrayLike,
                 sampling_rate: int,
                 window_len: int = 500,
                 window_edge: int = 40):
        '''
        Base model for all phase estimators. This class contains the helper
        functions, validation and base parameters needed for all estimators.

        Parameters
        ----------
        real_time_filter
            Filter parameters for filter to apply for predicting phase.
            A 1D FIR numerator, a 2-row (numerator, denominator) array, or
            an (n_sections, 6) array of second-order sections. Use
            second-order sections for IIR filters
        ground_truth_filter
            Filter parameters for identifying "true" phase. See
            :cite:t:Zrenner2020-zb for an in-depth discussion
            on "true" phase. Same formats as ``real_time_filter``
        sampling_rate : int
            Original sampling rate of data.
        window_len : int
            Window length in ms. Optional parameter to specify window length
            to run pseudo-real-time simulations or to train `window_len`
            dependent models with. This should match whatever is used in
            real-time
        window_edge : int
            Window edge to remove in ms. Optional parameter to specify edge to
            remove after applying `real_time_filter`
        '''

        _check_type(real_time_filter, ['array'])
        _check_type(ground_truth_filter, ['array'])
        _check_type(window_len, ['int'])
        _check_type(window_edge, ['int'])
        _check_type(sampling_rate, ['int'])

        _check_array_dimensions(real_time_filter, [(1,), (1, 1)])
        _check_array_dimensions(ground_truth_filter, [(1,), (1, 1)])

        # checks the shape (FIR, (b, a) or SOS) and that IIR filters are
        # stable, so a bad filter fails here instead of returning NaN phases
        _check_filter_coefficients(real_time_filter, 'real_time_filter')
        _check_filter_coefficients(ground_truth_filter, 'ground_truth_filter')

        # validate edge not 0
        if int(sampling_rate * (window_edge / 1000)) == 0:
            raise ValueError(
                "window_edge is too small for the given sampling rate. " +
                "Please increase window_edge or decrease sampling_rate")

        # validate 2*edge does not exceed window length
        if int(sampling_rate * (window_edge / 1000)) * 2 >= \
                int(sampling_rate * (window_len / 1000)):
            raise ValueError(
                "window_edge is too large for the given window_len. " +
                "Please decrease window_edge or increase window_len")

        self.ground_truth_filter: npt.ArrayLike = ground_truth_filter
        self.sampling_rate: int = sampling_rate

        self.real_time_filter: npt.ArrayLike = real_time_filter
        self.window_len: int = window_len
        self.window_edge: int = window_edge

        # (key, analytic signal) of the last ground-truth filtered recording
        self._ground_truth_cache = None

    def _filter_data(self, dsp_filter: npt.ArrayLike,
                     data: npt.ArrayLike) -> np.ndarray:
        '''
        Forward/backward filters data using an FIR or IIR filter

        Parameters
        ----------
        dsp_filter : array_like (n_parameters) | array_like (2, n_parameters)
                     | array_like (n_sections, 6)
            Filter to apply to data: an FIR numerator, a (numerator,
            denominator) pair or second-order sections
        data : array (n_samples,)
            1D array representing window to filter

        -------
        Returns
        -------
        filtered_data : array (n_samples,)
            Data after filtering
        '''
        coefficients = np.asarray(dsp_filter)
        form = _filter_form(coefficients)
        if form == 'sos':
            return signal.sosfiltfilt(coefficients, data)
        if form == 'ba':
            return signal.filtfilt(coefficients[0], coefficients[1], data)
        if form == 'fir':
            return signal.filtfilt(coefficients, 1.0, data)
        raise ValueError(
            'filter must be a 1D numerator or a 2-row (numerator, '
            'denominator) array, or an (n_sections, 6) array of '
            'second-order sections')

    def _ground_truth_analytic_signal(self,
                                      data: npt.ArrayLike) -> np.ndarray:
        '''
        Ground-truth filter `data` and return its analytic signal.

        The result for the most recent recording is cached, keyed on the
        contents of the data and of ``ground_truth_filter``, so computing
        several statistics or plots from the same recording filters it once.
        The returned array is read-only

        Parameters
        ----------
        data : array_like (n_samples,)
            Raw EEG data

        Returns
        -------
        analytic_signal : array (n_samples,)
            Hilbert transform of the ground-truth filtered data. Its real
            part is the filtered data and its angle is the phase in radians
        '''
        data = np.ascontiguousarray(data, dtype=float)
        gt_filter = np.ascontiguousarray(self.ground_truth_filter,
                                         dtype=float)

        digest = hashlib.blake2b(digest_size=16)
        digest.update(data)
        digest.update(gt_filter)
        key = (data.shape, gt_filter.shape, digest.digest())

        cache = getattr(self, '_ground_truth_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]

        analytic_signal = signal.hilbert(
            self._filter_data(gt_filter, data))
        analytic_signal.setflags(write=False)
        self._ground_truth_cache = (key, analytic_signal)

        return analytic_signal

    @staticmethod
    def _format_phase(phase: np.ndarray, degree: bool,
                      full_circle: bool = True) -> np.ndarray:
        '''
        Convert phases in radians in [-pi, pi) to the requested units and
        range
        '''
        if degree:
            phase = np.rad2deg(phase)
        if full_circle:
            phase = phase % 360 if degree else phase % (2 * np.pi)
        return phase

    @staticmethod
    def _circular_stats(phase: np.ndarray,
                        degree: bool) -> tuple:
        '''
        Circular mean and standard deviation of phases given in radians.
        Both are nan when there are no phases
        '''
        if len(phase) == 0:
            return np.nan, np.nan

        phase = phase % (2 * np.pi)
        mean = stats.circmean(phase)
        std = stats.circstd(phase)
        if degree:
            return float(np.rad2deg(mean) % 360), float(np.rad2deg(std))
        return float(mean), float(std)

    @staticmethod
    def _accuracy(phase: np.ndarray, target_phase: float) -> float:
        '''
        Accuracy of phases in radians to a target phase in degrees. nan when
        there are no phases
        '''
        if len(phase) == 0:
            return np.nan

        target_phase_rad = np.deg2rad(target_phase)
        phase_errors = np.angle(np.exp(1j * (phase - target_phase_rad)))
        mean_abs_error_deg = np.mean(np.abs(np.rad2deg(phase_errors)))

        return float(1 - mean_abs_error_deg / 180)

    def check_power_threshold(self,
                              data: npt.ArrayLike,
                              power_threshold: Union[int, float]) -> bool:
        '''
        Check whether the current window has sufficient power in the
        real-time filter's frequency band.

        Power is calculated by filtering the window, taking the Hilbert
        amplitude envelope, trimming the configured filter-edge samples from
        both ends, and averaging the squared envelope. The threshold is
        therefore expressed in squared input-amplitude units.

        Parameters
        ----------
        data : array_like (n_samples,)
            The unfiltered EEG data in the current window
        power_threshold : int | float
            Minimum mean-square Hilbert-envelope power required for the window

        Returns
        -------
        meets_threshold : bool
            Whether the window's power is greater than or equal to the
            threshold
        '''
        _check_type(data, ['array'])
        power_threshold = _check_real_number(
            power_threshold, 'power_threshold', non_negative=True)
        window = _check_real_array(data, 'data', contents='samples')
        _check_array_dimensions(window, [(1,)])

        edge = int((self.window_edge / 1000) * self.sampling_rate)
        (_, numerator_length,
         denominator_length) = _check_filter_coefficients(
            self.real_time_filter)

        filter_length = max(numerator_length, denominator_length)
        pad_length = 3 * filter_length
        minimum_window_length = max(pad_length + 1, 2 * edge + 1)
        if len(window) < minimum_window_length:
            raise ValueError(
                'data window is too short: expected at least '
                f'{minimum_window_length} samples for filtering and edge '
                f'removal, got {len(window)}')

        try:
            filtered_data = self._filter_data(
                self.real_time_filter, window)
        except (TypeError, ValueError) as error:
            raise ValueError(
                'real_time_filter could not filter the data window: '
                f'{error}') from error

        analytic_signal = signal.hilbert(filtered_data)
        envelope = np.abs(analytic_signal)[edge:-edge]
        if not np.all(np.isfinite(envelope)):
            raise ValueError(
                'real_time_filter produced non-finite values for this window')
        with np.errstate(over='ignore', invalid='ignore'):
            power = np.mean(np.square(envelope))
        if not np.isfinite(power):
            raise ValueError(
                'computed power is non-finite; check the signal scale and '
                'filter coefficients')

        return bool(power >= power_threshold)

    def get_phase_from_triggers(self,
                                data: npt.ArrayLike,
                                triggers: npt.ArrayLike,
                                degree: bool = True,
                                full_circle: bool = True) -> np.ndarray:
        '''
        Get the ground-truth phase at each trigger sample

        Parameters
        ----------
        data : array_like (n_samples)
            The raw (unfiltered) EEG data that the triggers correspond to.
            ``ground_truth_filter`` is applied by this method, so do not
            filter the data beforehand
        triggers : array_like (n_triggers)
            The sample number each trigger occurred at within the
            given EEG data. Must be whole numbers in the range
            [0, n_samples); negative indices are rejected
        degree=True : bool
            Whether to return the phase in degrees. If False, the phase is
            returned in radians
        full_circle=True : bool
            Whether to express phase values in full circle format (0 to 360
            or 0 to :math:`2\\pi`), matching the rest of the library, or in
            half circle format (-180 to 180 or :math:`-\\pi` to
            :math:`\\pi`)

        Returns
        -------
        phase_data : array (n_triggers)
            The phase each trigger in the `triggers` argument occurred at,
            in degrees in [0, 360) by default. 0 is a peak and 180 is a
            trough
        '''
        _check_type(data, ["array"])
        _check_type(triggers, ["array"])
        _check_type(degree, ["bool"])
        _check_type(full_circle, ["bool"])
        _check_array_dimensions(data, [(1,)])
        _check_array_dimensions(triggers, [(1,)])

        triggers = _check_triggers(triggers, len(data))
        analytic_signal = self._ground_truth_analytic_signal(data)
        phase = np.angle(analytic_signal[triggers])

        return self._format_phase(phase, degree, full_circle)

    def get_waveforms_from_triggers(self,
                                    data: npt.ArrayLike,
                                    triggers: npt.ArrayLike,
                                    tmin: Union[int, float],
                                    tmax: Union[int, float]) -> np.ndarray:
        '''
        Get the corresponding ground-truth filtered waveform window for each
        trigger sample. Triggers whose window would extend past either end
        of the recording are skipped

        Parameters
        ----------
        data : array_like (n_samples)
            The raw (unfiltered) EEG data that the triggers correspond to.
            ``ground_truth_filter`` is applied by this method, so do not
            filter the data beforehand
        triggers : array_like (n_triggers)
            The sample number each trigger occurred at within the given
            EEG data. Must be whole numbers in the range [0, n_samples)
        tmin : int | float
            The time in seconds before the trigger to include in the
            waveform window, as a positive number (e.g. 0.1 for 100 ms
            before the trigger). Unlike MNE's ``tmin``, this is not negative
        tmax : int | float
            The time in seconds after the trigger to include in the waveform
            window

        Returns
        -------
        waveform_data : list of array (n_samples)
            Each waveform, ``int(tmin * sampling_rate) +
            int(tmax * sampling_rate)`` samples long
        '''
        _check_type(data, ["array"])
        _check_type(triggers, ["array"])
        _check_type(tmin, ["int", "float"])
        _check_type(tmax, ["int", "float"])
        _check_array_dimensions(data, [(1,)])
        _check_array_dimensions(triggers, [(1,)])

        if tmin < 0:
            raise ValueError(
                f"tmin must be non-negative, got {tmin}. tmin is the time "
                "before the trigger as a positive number (e.g. tmin=0.1 for "
                "100 ms before the trigger), not MNE's negative tmin")
        if tmax < 0:
            raise ValueError(
                f"tmax must be non-negative, got {tmax}. tmax is the time "
                "after the trigger")

        fs = self.sampling_rate
        window_start_sample = int(tmin * fs)
        window_end_sample = int(tmax * fs)
        if window_start_sample + window_end_sample == 0:
            raise ValueError(
                "tmin and tmax are too small for the sampling rate: the "
                "waveform window would contain no samples")

        triggers = _check_triggers(triggers, len(data))
        filtered_data = self._ground_truth_analytic_signal(data).real

        waveform_windows = []
        for trigger_sample in triggers:
            start = trigger_sample - window_start_sample
            end = trigger_sample + window_end_sample
            if start >= 0 and end <= len(filtered_data):
                waveform_windows.append(filtered_data[start:end].copy())

        return waveform_windows

    def mean_phase_from_triggers(self,
                                 data: npt.ArrayLike,
                                 triggers: npt.ArrayLike,
                                 degree: bool = True) -> float:
        '''
        Get the circular mean for phase from trigger samples

        Parameters
        -----------
        data : array_like (n_samples)
            The raw (unfiltered) EEG data array
        triggers : array_like (n_triggers)
            The samples at which triggers occurred
        degree : bool
            Optional. By default is True. Whether to return the mean in
            degrees (0 to 360) or radians (0 to :math:`2\\pi`)

        Returns
        ---------
        circular_mean_phase : float
            The circular mean phase. nan if there are no triggers
        '''
        _check_type(degree, ["bool"])
        phase_data = self.get_phase_from_triggers(
            data, triggers, degree=False, full_circle=False)

        return self._circular_stats(phase_data, degree)[0]

    def std_phase_from_triggers(self,
                                data: npt.ArrayLike,
                                triggers: npt.ArrayLike,
                                degree: bool = True) -> float:
        '''
        Get the circular standard deviation for phase from trigger samples

        Parameters
        -----------
        data : array_like (n_samples)
            The raw (unfiltered) EEG data array
        triggers : array_like (n_triggers)
            The samples at which triggers occurred
        degree : bool
            Optional. By default is True. Whether to return the standard
            deviation in degrees or radians

        Returns
        --------
        phase_circular_std : float
            The phase data's circular standard deviation. nan if there are
            no triggers
        '''
        _check_type(degree, ["bool"])
        phase_data = self.get_phase_from_triggers(
            data, triggers, degree=False, full_circle=False)

        return self._circular_stats(phase_data, degree)[1]

    def phase_accuracy_from_triggers(self,
                                     data: npt.ArrayLike,
                                     triggers: npt.ArrayLike,
                                     target_phase: Union[int, float]) \
            -> float:
        '''
        Compute the accuracy of the triggers to the target phase. An accuracy
        of 50% means that the targeting is completely random. An accuracy
        below 50% means, the triggers tend to occur more often at the opposite
        phase.

        Parameters
        -----------
        data : array_like (n_samples)
            The raw (unfiltered) EEG data array
        triggers : array_like (n_triggers)
            The array containing the samples trigger occurred at
        target_phase : int | float
            The phase to target given in degrees

        Returns
        --------
        accuracy : float
            The accuracy in decimal form. nan if there are no triggers, since
            accuracy is undefined without them (0 would mean "perfectly
            anti-phase")
        '''
        target_phase = _check_real_number(target_phase, 'target_phase')
        phase = self.get_phase_from_triggers(
            data, triggers, degree=False, full_circle=False)

        return self._accuracy(phase, target_phase)

    def compute_phase_stats(self,
                            data: npt.ArrayLike,
                            triggers: npt.ArrayLike,
                            target_phase: Optional[Union[int, float]] = None,
                            degree: bool = True) -> dict:
        '''
        Compute the phase at each trigger, and the circular mean, circular
        standard deviation and accuracy of those phases, filtering the data
        only once.

        Parameters
        -----------
        data : array_like (n_samples)
            The raw (unfiltered) EEG data array
        triggers : array_like (n_triggers)
            The samples at which triggers occurred
        target_phase : None | int | float
            The phase that was targeted, in degrees. If None, accuracy is not
            computed
        degree : bool
            Optional. By default is True. Whether to return phases, mean and
            standard deviation in degrees (0 to 360) or radians
            (0 to :math:`2\\pi`)

        Returns
        --------
        phase_stats : dict
            ``'phase'``: array of the phase at each trigger, ``'mean'``: the
            circular mean, ``'std'``: the circular standard deviation and
            ``'accuracy'``: the accuracy to ``target_phase`` (None if no
            target phase was given). Mean, standard deviation and accuracy
            are nan if there are no triggers
        '''
        _check_type(degree, ["bool"])
        if target_phase is not None:
            target_phase = _check_real_number(target_phase, 'target_phase')

        phase = self.get_phase_from_triggers(
            data, triggers, degree=False, full_circle=False)
        mean, std = self._circular_stats(phase, degree)
        accuracy = None if target_phase is None \
            else self._accuracy(phase, target_phase)

        return {
            'phase': self._format_phase(phase, degree),
            'mean': mean,
            'std': std,
            'accuracy': accuracy,
        }

    def plot_mean_std_waveform_from_triggers(self,
                                             data: npt.ArrayLike,
                                             triggers: npt.ArrayLike,
                                             tmin: float,
                                             tmax: float) -> \
            matplotlib.pyplot.Figure:
        '''
        Plot mean waveform and standard deviation highlight from a set of
        triggers

        Parameters
        -----------
        data : array_like (n_samples)
            The raw EEG data array
        triggers : array_like (n_samples)
            The samples at which triggers occurred
        tmin : float
            The time in seconds before the trigger to include in the waveform
            window, as a positive number (e.g. 0.1 for 100 ms before). Unlike
            MNE's ``tmin``, this is not negative
        tmax : float
            The time in seconds post-trigger to include in the waveform window

        Returns
        --------
        waveform_avg_plot: matplotlib.pyplot.Figure
            The pyplot figure of the average waveform
        '''
        fs = self.sampling_rate
        waveforms = self.get_waveforms_from_triggers(
            data, triggers, tmin, tmax)

        if len(waveforms) == 0:
            raise ValueError(
                "No waveforms could be extracted for the provided triggers " +
                "and time window")

        # t_trigger is the sample index (within each extracted segment) that
        # corresponds to t=0
        t_trigger = int(fs * tmin)

        return plot_waveform_average(waveforms, fs, t_trigger)

    def polar_histogram_from_triggers(self,
                                      data: npt.ArrayLike,
                                      triggers: npt.ArrayLike) -> \
            matplotlib.pyplot.Figure:
        '''
        Plot polar histogram from trigger samples and raw EEG data

        Parameters
        -----------
        data : array_like (n_samples)
            The raw EEG data array
        triggers : array_like (n_samples)
            The samples at which triggers occurred

        Returns
        --------
        polar_histogram_figure : matplotlib.pyplot.Figure
            The pyplot figure of the polar histogram
        '''
        phase_data = self.get_phase_from_triggers(
            data, triggers, degree=False)

        return plot_polar_histogram(phase_data)
