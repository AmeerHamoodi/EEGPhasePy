import pytest 
import numpy as np
import scipy.signal as signal


from ..estimators import ETP

real_time_filter_fir = signal.firwin(128, [8, 13], fs=2048, pass_zero=False)
real_time_filter_iir = signal.butter(3, [8, 13], btype="bandpass", fs=2048)

ground_truth_filter_fir = signal.firwin(300, [8, 13], fs=2048, pass_zero=False)
ground_truth_filter_iir = signal.butter(3, [8, 13], btype="bandpass", fs=2048)

param_to_prop_map = ["real_time_filter", "ground_truth_filter", "sampling_rate", "window_len", "window_edge"]

def construct_etp_with_paramset(param_set):
  '''
  Constructs the ETP class with passed in parameters and returns the constructed ETP object

  Parameters
  ----------

  param_set : array
      Array of parameters in order to pass into ETP
  
  -------
  Returns
  -------

  ETP : the constructed ETP object  
  '''

  if len(param_set) == 3:
    return ETP(param_set[0], param_set[1], param_set[2])
  else:
    return ETP(param_set[0], param_set[1], param_set[2], window_len=param_set[3], window_edge=param_set[4])

def construct_default_etp():
  '''
  Constructs an ETP class with default parameters
  
  -------
  Returns
  -------

  etp : ETP object
  '''
  return ETP(real_time_filter_fir, ground_truth_filter_fir, 2048)

def test_etp_construct():
  # check ETP fails with TypeError if invalid type passed in
  # check ETP fails with ValueError if filter parameter array of wrong dimension
  # check ETP constructs correctly with the same values
  etp_parameter_sets = [
    [real_time_filter_fir, ground_truth_filter_fir, 2048], # pass
    [real_time_filter_iir, ground_truth_filter_iir, 2048], # pass
    [real_time_filter_fir, ground_truth_filter_fir, 2048.5], # fail
    [real_time_filter_fir, 1, 2048], # fail
    [1, ground_truth_filter_fir, 2048], # fail
    [real_time_filter_fir, ground_truth_filter_fir, 2048, 1000, 80], # pass
    [real_time_filter_fir, ground_truth_filter_fir, 2048, 1200.2, 60], # fail
    [real_time_filter_fir, ground_truth_filter_fir, 2048, 1000, 74.3], # fail
    [np.zeros((1, 1, 1)), ground_truth_filter_fir, 2048], # fail
    [real_time_filter_iir, np.zeros((1, 1, 1)), 2048], # fail
  ]
  etp_parameter_outputs = [
    False,
    False,
    [TypeError, "Value must be an int type"],
    [TypeError, "Value must be an array type"],
    [TypeError, "Value must be an array type"],
    False,
    [TypeError, "Value must be an int type"],
    [TypeError, "Value must be an int type"],
    [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D 2D"],
    [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D 2D"]
  ]
  
  for i, param_set in enumerate(etp_parameter_sets):
    if not type(etp_parameter_outputs[i]) == bool:
      with pytest.raises(etp_parameter_outputs[i][0], match=r"" + etp_parameter_outputs[i][1] +""):
        construct_etp_with_paramset(param_set)
    else:
      etp = construct_etp_with_paramset(param_set)
      for j, prop in enumerate(param_to_prop_map[:len(param_set)]):
        if hasattr(param_set[j], '__len__'):
          assert (np.array(getattr(etp, prop)) == np.array(param_set[j])).all()
        else:
          assert getattr(etp, prop) == param_set[j]

def test_etp_fit():
  # should fail if training data not 1D
  # should fail if min_ipi not int
  # should fail if training_data not arr

  etp_fit_parameters = [
    [np.sin(78.5*np.arange(0, 200, 1/2048)), 63], # pass
    [1, 63], # fail
    [np.zeros((100, 2)), 63], # fail
    [np.zeros(100), 62.5] # fail
  ]

  etp_fit_parameter_outputs = [
    False,
    [TypeError, "Value must be an array type"],
    [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
    [TypeError, "Value must be an int type"]
  ]

  for i, param_set in enumerate(etp_fit_parameters):
    if not type(etp_fit_parameter_outputs[i]) == bool:
      with pytest.raises(etp_fit_parameter_outputs[i][0], match=r"" + etp_fit_parameter_outputs[i][1] +""):
        etp = construct_default_etp()
        etp.fit(param_set[0], param_set[1])
    else:
      etp = construct_default_etp()
      etp.fit(param_set[0], param_set[1])
      assert etp.Tadj == pytest.approx(80, 2) # check if Tadj roughly equal to pre-defined period (80ms)