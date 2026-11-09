import pytest
import numpy as np
import scipy.signal as signal
import scipy.stats as stats

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
        [np.sin(78.5*np.arange(0, 500, 1/2048)), 63],  # pass
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
            assert etp.tadj == pytest.approx(int(0.08 * 2048), abs=10)


def test_etp_predict():
    # Test if predict successfuly predicts with no errors
    # Test if predict fails when given non-array
    # Test if predict fails when given 2D array
    # Test if predict fails when given non-float target_phase

    etp_predict_parameters = [
        [np.sin(78.5*np.arange(0, 0.5, 1/2048)), 0],
        [15, 0],
        [np.zeros((500, 2)), 0],
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

            samples_to_wait = etp.predict(param_set[0], param_set[1])
            # predict returns the number of samples to wait after the end of the
            # window before delivering a stimulus
            assert isinstance(samples_to_wait, np.int64)
            assert 0 <= samples_to_wait <= etp.tadj


@pytest.mark.parametrize("target_phase", [0, 90, 180, 270])
def test_etp_real_data(resting_state_eeg, target_phase):
    training_data, testing_data, fs = resting_state_eeg

    etp = ETP(signal.firwin(128, [8, 12], fs=fs, pass_zero=False),
              signal.firwin(500, [8, 12], fs=fs, pass_zero=False),
              fs, window_len=2000)
    etp.fit(training_data, min_ipi=int(fs / 12))

    window_len = int(2 * fs)
    window_step = int(0.06 * fs)

    triggers = []
    for window_i in range(0, len(testing_data) - window_len, window_step):
        window_data = testing_data[window_i:window_i + window_len]
        try:
            trigger = window_i + window_len + \
                etp.predict(window_data, target_phase)
        except RuntimeError:
            continue  # no peaks found in this window

        if trigger < len(testing_data):
            triggers.append(trigger)

    mean_phase = etp.mean_phase_from_triggers(
        testing_data, triggers, degree=True)
    accuracy = etp.phase_accuracy_from_triggers(
        testing_data, triggers, target_phase)

    assert len(triggers) > 1000
    assert abs((mean_phase - target_phase + 180) % 360 - 180) < 25
    assert accuracy > 0.65


@pytest.fixture(scope="module")
def fitted_etp():
    etp = construct_default_etp()
    etp.fit(np.sin(78.5*np.arange(0, 200, 1/2048)), 63)
    return etp


@pytest.fixture(scope="module")
def noisy_windows():
    fs = 2048
    rng = np.random.default_rng(0)
    time_data = np.arange(0, 20, 1/fs)
    noisy = np.sin(78.5 * time_data) + rng.normal(0, 0.5, len(time_data))
    return [noisy[i:i + 1024] for i in range(0, len(noisy) - 1024, 307)]


def test_etp_predict_before_fit():
    with pytest.raises(RuntimeError, match=r"ETP has not been fit"):
        construct_default_etp().predict(np.sin(78.5*np.arange(0, 0.5, 1/2048)), 0)


def test_etp_predict_no_peaks_message(fitted_etp):
    with pytest.raises(RuntimeError) as error:
        fitted_etp.predict(np.zeros(1024), 0)
    assert str(error.value) == \
        "No peaks could be found in the window passed into the `predict` method"


def test_etp_predict_wraps_target_phase(fitted_etp, noisy_windows):
    for window in noisy_windows:
        assert fitted_etp.predict(window, -90) == fitted_etp.predict(window, 270)
        assert fitted_etp.predict(window, 450) == fitted_etp.predict(window, 90)
        assert fitted_etp.predict(window, -360) == fitted_etp.predict(window, 0)


@pytest.mark.parametrize("target_phase", [0, 45, 90, 180, 270, 359])
def test_etp_predict_never_in_past(fitted_etp, noisy_windows, target_phase):
    for window in noisy_windows:
        samples_to_wait = fitted_etp.predict(window, target_phase)
        assert 0 <= samples_to_wait < fitted_etp.period

        delayed = fitted_etp.predict(window, target_phase, 300)
        assert 300 <= delayed < 300 + fitted_etp.period
        # the delayed trigger is the same phase, whole periods later
        assert (delayed - samples_to_wait) % fitted_etp.period == 0


def test_etp_predict_min_samples_to_wait_validation(fitted_etp):
    window = np.sin(78.5*np.arange(0, 0.5, 1/2048))
    with pytest.raises(ValueError, match=r"min_samples_to_wait must be non-negative"):
        fitted_etp.predict(window, 0, -1)
    with pytest.raises(TypeError, match=r"Value must be an int type"):
        fitted_etp.predict(window, 0, 1.5)


def test_etp_predict_applies_tadj_to_every_target(noisy_windows):
    etp = construct_default_etp()
    etp.fit(np.sin(78.5*np.arange(0, 200, 1/2048)), 63)
    window = noisy_windows[0]

    for target_phase in [0, 90, 180, 270]:
        expected_step = round(etp.period * target_phase / 360)
        assert (etp.predict(window, target_phase) - etp.predict(window, 0)
                - expected_step) % etp.period == 0

    # a bias correction in tadj shifts every target by the same amount
    before = {phase: etp.predict(window, phase) for phase in [0, 90, 180]}
    etp.tadj += 10
    for phase, samples_to_wait in before.items():
        assert (etp.predict(window, phase) - samples_to_wait - 10) \
            % etp.period == 0


def test_etp_fit_too_short():
    etp = construct_default_etp()
    short_signal = np.sin(78.5*np.arange(0, 120, 1/2048))

    with pytest.raises(ValueError, match=r"training_data is too short"):
        etp.fit(short_signal, 63)

    etp.fit(short_signal, 63, ipi_duration=30, n_windows=100)
    assert etp.tadj == pytest.approx(int(0.08 * 2048), abs=10)


@pytest.mark.parametrize("kwargs, error, message", [
    ({"ipi_duration": 0}, ValueError, "ipi_duration must be positive"),
    ({"ipi_duration": -1}, ValueError, "ipi_duration must be non-negative"),
    ({"ipi_duration": "90"}, TypeError, "ipi_duration must be a real number"),
    ({"n_windows": 0}, ValueError, "n_windows must be at least 1"),
    ({"n_windows": 2.5}, TypeError, "Value must be an int type"),
])
def test_etp_fit_layout_validation(kwargs, error, message):
    with pytest.raises(error, match=message):
        construct_default_etp().fit(
            np.sin(78.5*np.arange(0, 200, 1/2048)), 63, **kwargs)


def test_etp_fit_no_valid_ipi():
    with pytest.raises(ValueError, match=r"No inter-peak intervals of at least min_ipi"):
        construct_default_etp().fit(
            np.sin(78.5*np.arange(0, 200, 1/2048)), 10000)


def test_etp_fit_min_ipi_is_inclusive():
    fs = 2000
    etp = ETP(signal.firwin(120, [8, 12], fs=fs, pass_zero=False),
              signal.firwin(300, [8, 12], fs=fs, pass_zero=False), fs)
    # 10 Hz at 2000 Hz gives inter-peak intervals of exactly 200 samples
    etp.fit(np.sin(2 * np.pi * 10 * np.arange(0, 200, 1/fs)), 200)
    assert etp.period == 200
