import pytest
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import mne
import os


from EEGPhasePy.estimators import ETP

real_time_filter_fir = signal.firwin(128, [8, 13], fs=2048, pass_zero=False)
real_time_filter_iir = signal.butter(3, [8, 13], btype="bandpass", fs=2048)

ground_truth_filter_fir = signal.firwin(300, [8, 13], fs=2048, pass_zero=False)
ground_truth_filter_iir = signal.butter(3, [8, 13], btype="bandpass", fs=2048)

param_to_prop_map = ["real_time_filter", "ground_truth_filter",
                     "sampling_rate", "window_len", "window_edge"]


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


def test_etp_fit():
    # should fail if training data not 1D
    # should fail if min_ipi not int
    # should fail if training_data not arr

    etp_fit_parameters = [
        [np.sin(78.5*np.arange(0, 200, 1/2048)), 63],  # pass
        [1, 63],  # fail
        [np.zeros((100, 2)), 63],  # fail
        [np.zeros(100), 62.5]  # fail
    ]

    etp_fit_parameter_outputs = [
        False,
        [TypeError, "Value must be an array type"],
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [TypeError, "Value must be an int type"]
    ]

    for i, param_set in enumerate(etp_fit_parameters):
        if not type(etp_fit_parameter_outputs[i]) == bool:
            with pytest.raises(etp_fit_parameter_outputs[i][0], match=r"" + etp_fit_parameter_outputs[i][1] + ""):
                etp = construct_default_etp()
                etp.fit(param_set[0], param_set[1])
        else:
            etp = construct_default_etp()
            etp.fit(param_set[0], param_set[1])
            # check if Tadj roughly equal to pre-defined period (80ms)
            assert etp.Tadj == pytest.approx(80, 2)


def test_etp_predict():
    # Test if predict successfuly predicts with no errors
    # Test if predict fails when given non-array
    # Test if predict fails when given 2D array
    # Test if predict fails when given non-float target_phase

    etp_predict_parameters = [
        [np.sin(78.5*np.arange(0, 0.5, 1/2048)), 1.96],
        [15, 1.96],
        [np.zeros((500, 2)), 1.96],
        [np.sin(78.5*np.arange(0, 0.5, 1/2048)), "string"]
    ]

    etp_predict_parameter_outputs = [
        False,
        [TypeError, "Value must be an array type"],
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [TypeError, "Value must be one of: float or int"]
    ]

    for i, param_set in enumerate(etp_predict_parameters):
        if not type(etp_predict_parameter_outputs[i]) == bool:
            with pytest.raises(etp_predict_parameter_outputs[i][0], match=r"" + etp_predict_parameter_outputs[i][1] + ""):
                etp = construct_default_etp()
                etp.fit(np.sin(78.5*np.arange(0, 200, 1/2048)), 63)
                etp.predict(param_set[0], param_set[1])
        else:
            etp = construct_default_etp()
            etp.fit(np.sin(78.5*np.arange(0, 200, 1/2048)), 63)

            assert isinstance(etp.predict(
                param_set[0], param_set[1]), np.int64)


def test_etp_real_data():
    if os.getenv('GITHUB_ACTIONS') == True:
        return
    # Load test files
    training_data = mne.io.read_raw_curry(
        "./tests/test_data/training-rsEEG.cdt")
    testing_data = mne.io.read_raw_curry("./tests/test_data/testing-rsEEG.cdt")
    fs = 2048

    C3_train_data = training_data.pick("C3").get_data()[0]
    C3_test_data = testing_data.pick("C3").get_data()[0]

    # Test if predict performs as expected on test files mean == mean and std == std for phase

    etp = construct_default_etp()
    etp.fit(C3_train_data, 83)

    window_i = 0
    window_step = int(0.063 * fs)
    window_len = int(0.5 * fs)

    triggers = []

    while window_i + window_len < len(C3_test_data):
        window_data = C3_test_data[window_i:window_i + window_len]

        triggers.append(etp.predict(window_data, 0) + window_i)

        window_i += window_step

    # Obtain phase from trigger indecies
    assert stats.circmean(etp.get_phase_from_triggers(C3_test_data, triggers, toDegree=True, fullCircle=True)) == pytest.approx(0, 5) \
        or stats.circmean(etp.get_phase_from_triggers(C3_test_data, triggers, toDegree=True, fullCircle=True)) == pytest.approx(360, 5)
