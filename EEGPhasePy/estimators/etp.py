import numpy as np
import scipy.signal as signal
import scipy.stats as stats

from ..utils.check import _check_array_dimensions, _check_type

class ETP:
  def __init__(self, real_time_filter, ground_truth_filter, sampling_rate, window_len=500, window_edge=40):
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
    self.Tadj = None

    _check_type(real_time_filter, 'array')
    _check_type(ground_truth_filter, 'array')
    _check_type(window_len, 'int')
    _check_type(window_edge, 'int')
    _check_type(sampling_rate, 'int')

    _check_array_dimensions(real_time_filter, [(1,), (1, 1)])
    _check_array_dimensions(ground_truth_filter, [(1,), (1,1)])

    self.ground_truth_filter = ground_truth_filter
    self.sampling_rate = sampling_rate
    
    self.real_time_filter = real_time_filter
    self.window_len = window_len
    self.window_edge = window_edge
  
  def _filter_data(self, dsp_filter, data):
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
  
  def fit(self, training_data, min_ipi):
    '''
    Estimates ideal Tadj for training data and updates self object with new Tadj.

    Parameters
    ----------
    training_data : array (n_samples,)
        1D samples array of channel to estimate Tadj for
    min_ipi : int
        Minimum inter peak interval, should be the period of the upper frequency of the target band
    
    -------
    Returns
    -------
    self
    '''
    _check_type(training_data, "array")
    _check_type(min_ipi, "int")
    _check_array_dimensions(training_data, [(1,)]) # ensure training data is 1D
        
    fs = 1000
    resampled_training_data = signal.resample(training_data, int(fs*len(training_data) / self.sampling_rate))
    filtered_data = self._filter_data(self.ground_truth_filter, resampled_training_data)
    # ground truth is hard to define for phase estimation, see Zrenner et al., 2020 for a more detailed discussion
    ground_truth_phase = np.angle(signal.hilbert(filtered_data), deg=True) % 360

    training_window_data = filtered_data[:90*fs]
    peaks = signal.find_peaks(training_window_data)[0]
    inter_peak_interval = np.diff(peaks)
    inter_peak_interval = inter_peak_interval[inter_peak_interval > min_ipi] # remove IPIs that are too short
    period = round(np.exp(np.nanmean(np.log(inter_peak_interval))))

    bias = 0
    bias_direction = None
    last_mean = None
    mean_differences = []
    while True:
      triggered_phases = []

      for i in range(255):
        window_i = 90*fs + 350*i
        window_data = resampled_training_data[window_i:window_i + self.window_len]
        filtered_window = self._filter_data(self.real_time_filter, window_data)[:-self.window_edge]

        peaks = signal.find_peaks(filtered_window)[0]
        trigger_i = window_i + peaks[-1] + period + bias
        triggered_phases.append(ground_truth_phase[trigger_i])
      
      mean_phase = stats.circmean(np.deg2rad(triggered_phases))
      
      if bias_direction == None:
        bias_direction = 1 if np.rad2deg(mean_phase)%360 > 180 else -1
      
      if not last_mean == None:
        difference_mean = 1 - np.real(np.exp(1j*last_mean - 1j*mean_phase))
        if len(mean_differences) > 0 and mean_differences[-1] < difference_mean:
          self.Tadj = period + bias - bias_direction
          return self
        
        mean_differences.append(difference_mean)
      last_mean = mean_phase
      
      bias += bias_direction

  def predict(self, data, target_phase):
    '''
    Predicts the next sample target phase occurs at

    Parameters
    ----------
    data : array_like (n_samples)
        Window of EEG data from target channel to predict from
    target_phase : float
        Target phase to predict in radians

    -------
    Returns
    -------
    relative_next_phase : int
        Next sample target phase occurs, defined relative to window start
    '''

    _check_type(data, "array")
    _check_type(target_phase, "float")

    _check_array_dimensions(data, [(1,)])

    fs = 1000
    downsampled_window = signal.resample(data, int(fs*len(data) / self.sampling_rate))
    filtered_window = self._filter_data(self.real_time_filter, downsampled_window)
    peaks = signal.find_peaks(filtered_window)[0]

    if len(peaks) == 0:
      raise RuntimeError("No peaks could be found in the window passed into the `predict` method")

    Tadj = self.Tadj * target_phase/2*np.pi

    return peaks[-1] + Tadj
    
