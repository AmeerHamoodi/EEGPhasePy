from EEGPhasePy.estimators import Estimator
import pytest
import numpy as np
import scipy.signal as signal

from EEGPhasePy.utils import check

real_time_filter_fir = signal.firwin(128, [8, 13], fs=2048, pass_zero=False)
real_time_filter_iir = signal.butter(3, [8, 13], btype="bandpass", fs=2048)

ground_truth_filter_fir = signal.firwin(300, [8, 13], fs=2048, pass_zero=False)
ground_truth_filter_iir = signal.butter(3, [8, 13], btype="bandpass", fs=2048)

param_to_prop_map = ["real_time_filter", "ground_truth_filter",
                     "sampling_rate", "window_len", "window_edge"]


def construct_estimator_with_paramset(param_set):
    '''
    Constructs the Estimator class with passed in parameters and returns the constructed Estimator object

    Parameters
    ----------

    param_set : array
        Array of parameters in order to pass into Estimator

    -------
    Returns
    -------

    estimator : the constructed Estimator object  
    '''

    if len(param_set) == 3:
        return Estimator(param_set[0], param_set[1], param_set[2])
    else:
        return Estimator(param_set[0], param_set[1], param_set[2], window_len=param_set[3], window_edge=param_set[4])


def test_estimator_construct():
    estimator_parameter_sets = [
        [real_time_filter_fir, ground_truth_filter_fir, 2048],  # pass
        [real_time_filter_iir, ground_truth_filter_iir, 2048],  # pass
        [real_time_filter_fir, ground_truth_filter_fir, 2048.5],  # fail
        [real_time_filter_fir, 1, 2048],  # fail
        [1, ground_truth_filter_fir, 2048],  # fail
        [real_time_filter_fir, ground_truth_filter_fir, 2048, 1000, 80],  # pass
        [real_time_filter_fir, ground_truth_filter_fir, 2048, 1200.2, 60],  # fail
        [real_time_filter_fir, ground_truth_filter_fir, 2048, 1000, 74.3],  # fail
        [np.zeros((1, 1, 1)), ground_truth_filter_fir, 2048],  # fail
        [real_time_filter_iir, np.zeros((1, 1, 1)), 2048],  # fail
    ]
    estimator_parameter_outputs = [
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

    for i, param_set in enumerate(estimator_parameter_sets):
        if not type(estimator_parameter_outputs[i]) == bool:
            with pytest.raises(estimator_parameter_outputs[i][0], match=r"" + estimator_parameter_outputs[i][1] + ""):
                construct_estimator_with_paramset(param_set)
        else:
            etp = construct_estimator_with_paramset(param_set)
            for j, prop in enumerate(param_to_prop_map[:len(param_set)]):
                if hasattr(param_set[j], '__len__'):
                    assert (np.array(getattr(etp, prop)) ==
                            np.array(param_set[j])).all()
                else:
                    assert getattr(etp, prop) == param_set[j]


def test_estimator_get_phase_at_triggers():
    trigger_param_sets = [
        [np.zeros((1000, 2)), [1, 2, 3, 4]],
        [np.zeros(1000), [[1, 2, 3, 4]]],
        [1, [1, 2, 3, 4]],
        [np.zeros(1000), 1],
        [np.zeros(1000), [1, 2, 3, 4], 1],
        [np.zeros(1000), [1, 2, 3, 4], True],
        [np.zeros(1000), [1, 2, 3, 4], True, 1],
        [np.zeros(1000), [1, 2, 3, 4], True, True],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)), [125, 225]],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)), [238, 338, 438], True],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)),
         [238, 121, 221], True, True],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)),
         [238, 121, 221], False, True],
    ]

    trigger_param_outputs = [
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [TypeError, "Value must be an array"],
        [TypeError, "Value must be an array"],
        [TypeError, "Value must be a bool"],
        False,
        [TypeError, "Value must be a bool"],
        False,
        [0, 0],
        [45, 45, 45],
        [45, 345, 345],
        [0.25*np.pi, 6.02, 6.02]
    ]

    custom_filter = signal.firwin(100, [1, 5], fs=500, pass_zero=False)
    estimator = Estimator(custom_filter, custom_filter, 500)
    for i, param_set in enumerate(trigger_param_sets):
        if check._is_array(trigger_param_outputs[i]) and (trigger_param_outputs[i][0] == ValueError or trigger_param_outputs[i][0] == TypeError):
            with pytest.raises(trigger_param_outputs[i][0], match=r"" + trigger_param_outputs[i][1] + ""):
                phase = estimator.get_phase_from_triggers(*param_set)
        elif check._is_array(trigger_param_outputs[i]):
            phase = estimator.get_phase_from_triggers(*param_set)

            for j, true_phase in enumerate(trigger_param_outputs[i]):
                if true_phase <= 2*np.pi:
                    assert np.abs(phase[j]) == pytest.approx(
                        true_phase, abs=0.1)
                else:
                    assert np.abs(phase[j]) == pytest.approx(true_phase, abs=6)
        else:
            phase = estimator.get_phase_from_triggers(*param_set)


def test_estimator_compute_phase_stats():
    phase_param_sets = [
        [np.sin(2 * np.pi * 5 * np.arange(0, 10, 1/500)),
         [75, 175, 275, 375, 475, 575]],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)) +
         np.random.normal(0, 1, size=1000), [75, 175, 275]],
    ]

    trigger_param_outputs = [
        [180, 0],
        [180, 1]
    ]

    custom_filter = signal.firwin(100, [1, 5], fs=500, pass_zero=False)
    estimator = Estimator(custom_filter, custom_filter, 500)
    for i, param_set in enumerate(phase_param_sets):
        assert estimator.mean_phase_from_triggers(
            param_set[0], param_set[1], degree=True) == pytest.approx(trigger_param_outputs[i][0], abs=6)
        assert estimator.std_phase_from_triggers(
            param_set[0], param_set[1], degree=True) == pytest.approx(trigger_param_outputs[i][1], abs=15)
