import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import scipy.stats as stats
import matplotlib
from typing import Union

from ..utils.check import _check_array_dimensions, _check_type
from ..viz import plot_polar_histogram, plot_waveform_average


class Estimator:
    '''
      Base model for all phase estimators. This class contains the helper
      functions, validation and base parameters needed for all estimators.
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
            Accounts for FIR or IIR filters
        ground_truth_filter
            Filter parameters for identifying "true" phase. See
            :cite:t:Zrenner2020-zb for an in-depth discussion
            on "true" phase. Accounts for FIR or IIR filters
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

        self.ground_truth_filter: npt.ArrayLike = ground_truth_filter
        self.sampling_rate: int = sampling_rate

        self.real_time_filter: npt.ArrayLike = real_time_filter
        self.window_len: int = window_len
        self.window_edge: int = window_edge

    def _filter_data(self, dsp_filter: npt.ArrayLike,
                     data: npt.ArrayLike) -> np.ndarray:
        '''
        Forward/backward filters data using an FIR or IIR filter

        Parameters
        ----------
        dsp_filter : array_like (n_parameters) | array_like (2, n_parameters)
            Filter to apply to data
        data : array (n_samples,)
            1D array repreenting window to filter

        -------
        Returns
        -------
        filtered_data : array (n_samples,)
            Data after filtering
        '''
        return signal.filtfilt(dsp_filter, 1.0, data) \
            if len(np.shape(dsp_filter)) == 1 \
            else signal.filtfilt(dsp_filter[0], dsp_filter[1], data)

    def get_phase_from_triggers(self,
                                data: npt.ArrayLike,
                                triggers: npt.ArrayLike,
                                degree=False,
                                full_circle=False) -> np.ndarray:
        '''
        Get the corresponding phase for each trigger sample

        Parameters
        ----------
        data : array_like (n_samples)
            The ground truth filtered EEG data in array format that the
            triggers correspond to
        triggers : array_like (n_samples)
            The sample number each trigger occurred at within the
            given EEG data
        degree=False : bool
            Whether to convert the phase data into degree. By default this is
            value is false
        full_circle=False : bool
            Whether to express phase values in full circle format (i.e. 0 to
            360 or 0 to :math:`2\\pi`) or the default format
            (-180 to 180 or :math:r`-\\pi` to :math:`\\pi`)


        Returns
        -------
        phase_data : array_like (n_samples)
            An array containing the phase, in radians, each trigger in the
            `triggers` argument occurred at
        '''
        _check_type(data, ["array"])
        _check_type(triggers, ["array"])
        _check_type(degree, ["bool"])
        _check_type(full_circle, ["bool"])
        _check_array_dimensions(data, [(1,)])
        _check_array_dimensions(triggers, [(1,)])

        filtered_data = self._filter_data(self.ground_truth_filter, data)
        phase = np.angle(signal.hilbert(filtered_data)[triggers])

        if degree:
            phase = np.rad2deg(phase)

        if full_circle:
            phase = phase % 360 if degree else phase % (2*np.pi)

        return phase

    def get_waveforms_from_triggers(self,
                                    data: npt.ArrayLike,
                                    triggers: npt.ArrayLike,
                                    tmin: Union[int, float],
                                    tmax: Union[int, float]) -> np.ndarray:
        '''
        Get the corresponding waveform window for each trigger sample

        Parameters
        ----------
        data : array_like (n_samples)
            The ground truth filtered EEG data in array format that the
            triggers correspond to
        triggers : array_like (n_samples)
            The sample number each trigger occurred at within the given
            EEG data
        tmin : int | float
            The time in seconds pre-trigger to include in the waveform window
        tmax : int | float
            The time in seconds post-trigger to include in the waveform window

        Returns
        -------
        waveform_data : array_like (n_waveforms, n_samples)
            A 2D array containing the each waveform
        '''
        _check_type(data, ["array"])
        _check_type(triggers, ["array"])
        _check_type(tmin, ["int", "float"])
        _check_type(tmax, ["int", "float"])
        _check_array_dimensions(data, [(1,)])
        _check_array_dimensions(triggers, [(1,)])

        fs = self.sampling_rate
        waveform_windows = []
        window_start_sample = int(tmin * fs)
        window_end_sample = int(tmax * fs)

        filtered_data = self._filter_data(self.ground_truth_filter, data)

        for trigger_sample in triggers:
            waveform_windows.append(
                filtered_data[trigger_sample - window_start_sample:
                              trigger_sample + window_end_sample])

        return waveform_windows

    def mean_phase_from_triggers(self,
                                 data: npt.ArrayLike,
                                 triggers: npt.ArrayLike,
                                 degree=False) -> float:
        '''
        Get the circular mean for phase from trigger samples

        Parameters
        -----------
        data : array_like (n_samples)
            The raw EEG data array
        triggers : array_like (n_samples)
            The samples at which triggers occurred
        degree : bool
            Optional. By default is False. Whether to plot the polar histogram
            in degrees or radians

        Returns
        ---------
        circular_mean_phase : float
            The circular mean phase
        '''
        phase_data = self.get_phase_from_triggers(
            data, triggers, degree=False)

        if degree:
            return stats.circmean(np.rad2deg(phase_data) % 360)
        else:
            return stats.circmean(phase_data % 2*np.pi)

    def std_phase_from_triggers(self,
                                data: npt.ArrayLike,
                                triggers: npt.ArrayLike,
                                degree=False) -> float:
        '''
        Get the circular standard deviation for phase from trigger samples

        Parameters
        -----------
        data : array_like (n_samples)
            The raw EEG data array
        triggers : array_like (n_samples)
            The samples at which triggers occurred
        degree : bool
            Optional. By default is False. Whether to plot the polar histogram
            in degrees or radians

        Returns
        --------
        phase_circular_std : float
            The phase data's circular standard deviation
        '''
        phase_data = self.get_phase_from_triggers(
            data, triggers, degree=False)

        if degree:
            return stats.circstd(np.rad2deg(phase_data) % 360)
        else:
            return stats.circstd(phase_data % 2*np.pi)

    def phase_accuracy_from_triggers(self,
                                     data: npt.ArrayLike,
                                     triggers: npt.ArrayLike,
                                     target_phase: float) -> float:
        '''
        Compute the accuracy of the triggers to the target phase. An accuracy
        of 50% means that the targeting is completely random. An accuracy
        below 50% means, the triggers tend to occur more often at the opposite
        phase.

        Parameters
        -----------
        data : array_like (n_samples)
            The raw EEG data array
        triggers : array_like (n_samples)
            The array containing the samples trigger occurred at
        target_phase : float
            The phase to target given in radians

        Returns
        --------
        accuracy : float
            The accuracy in decimal form
        '''
        if len(triggers) == 0:
            return 0

        phase = self.get_phase_from_triggers(data, triggers)

        phase_difference = np.exp(phase*1j - target_phase*1j)
        mean_degree_difference = np.abs(np.angle(
            np.sum(phase_difference), deg=True)) / (len(triggers) * 180)

        return 1 - mean_degree_difference

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
            The time in seconds pre-trigger to include in the waveform window
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

        # Need to find t_trigger
        t_trigger = len(waveforms[0]) - (int(fs * tmin))

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
