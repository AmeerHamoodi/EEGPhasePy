import numpy as np
import scipy.signal as signal
import scipy.stats as stats

from ..utils.check import _check_array_dimensions, _check_type

class Estimator:
  def __init__(self, 
                real_time_filter: np.ndarray, 
                ground_truth_filter: np.ndarray, 
                sampling_rate: int, 
                window_len=500,
                window_edge=40):
    '''
    Base model for all phase estimators. This class contains the helper functions, validation and base parameters
    needed for all estimators.

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

    _check_type(real_time_filter, ['array'])
    _check_type(ground_truth_filter, ['array'])
    _check_type(window_len, ['int'])
    _check_type(window_edge, ['int'])
    _check_type(sampling_rate, ['int'])

    _check_array_dimensions(real_time_filter, [(1,), (1, 1)])
    _check_array_dimensions(ground_truth_filter, [(1,), (1,1)])

    self.ground_truth_filter: np.ndarray = ground_truth_filter
    self.sampling_rate: int = sampling_rate
    
    self.real_time_filter: np.ndarray = real_time_filter
    self.window_len: int = window_len
    self.window_edge: int = window_edge


  def _filter_data(self, dsp_filter: np.ndarray | list, data: np.ndarray | list) -> np.ndarray:
    '''
    Forward/backward filters data using an FIR or IIR filter

    Parameters
    ----------
    dsp_filter : array_like shape (n_parameters) | array_like shape (2, n_parameters)
        Filter to apply to data
    data : array (n_samples,)
        1D array repreenting window to filter
    
    -------
    Returns
    -------
    filtered_data : array (n_samples,)
        Data after filtering
    '''
    return signal.filtfilt(dsp_filter, 1.0, data) if hasattr(self.ground_truth_filter, '__len__') \
      else signal.filtfilt(dsp_filter[0], dsp_filter[0], data)

  def get_phase_from_triggers(self, data: np.ndarray, triggers: list | np.ndarray, toDegree=False, fullCircle=False) -> np.ndarray:
    '''
    Get the corresponding phase for each trigger sample

    Parameters
    ----------
    data : array_like (n_parameters)
        The ground truth filtered EEG data in array format that the triggers correspond to
    triggers : array_like (n_parameters)
        The sample number each trigger occurred at within the given EEG data
    toDegree=False : bool 
        Whether to convert the phase data into degree. By default this is value is false
    fullCircle=False : bool
        Whether to express phase values in full circle format (i.e. 0 to 360 or 0 to :math:`2r'\pi'`) or the default format (-180 to 180 or :math:`-r'\pi'` to :math:`r'\pi'`)
    -------
    Returns
    -------
    phase_data : array_like (n_parameters)
        An array containing the phase, in radians, each trigger in the `triggers` argument occurred at
    '''
    _check_type(data, ["array"])
    _check_type(triggers, ["array"])
    _check_array_dimensions(data, [(1,)])

    filtered_data = self._filter_data(self.ground_truth_filter, data)
    phase = np.angle(signal.hilbert(filtered_data)[triggers])

    if toDegree:
      phase = np.rad2deg(phase)
    
    if fullCircle:
      phase = phase%360 if toDegree else phase%2*np.pi

    return phase
