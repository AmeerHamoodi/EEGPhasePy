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
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)), [125, 225], False,
         False],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)), [238, 338, 438], True],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)),
         [238, 121, 221], True, True],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)),
         [238, 121, 221], False, True],
    ]

    trigger_param_outputs = [
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [ValueError, "The provided array has the wrong dimension. Arrays can have the following dimensions: 1D"],
        [TypeError, "Value must be an array type"],
        [TypeError, "Value must be an array type"],
        [TypeError, "Value must be a bool type"],
        False,
        [TypeError, "Value must be a bool type"],
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
    # local generator so the noise does not depend on test ordering. Seed 123
    # reproduces the draw this test was written against
    rng = np.random.RandomState(123)
    phase_param_sets = [
        [np.sin(2 * np.pi * 5 * np.arange(0, 10, 1/500)),
         [75, 175, 275, 375, 475, 575]],
        [np.sin(2 * np.pi * 5 * np.arange(0, 2, 1/500)) +
         rng.normal(0, 1, size=1000), [75, 175, 275]],
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


FS = 2048
TIME = np.arange(0, 10, 1 / FS)
ALPHA = np.sin(2 * np.pi * 10 * TIME)
SOS_6TH_ORDER = signal.butter(6, [8, 12], btype='bandpass', fs=FS,
                              output='sos')


def test_estimator_accepts_sos_filters():
    estimator = Estimator(real_time_filter_fir, SOS_6TH_ORDER, FS)
    triggers = np.arange(FS, len(ALPHA) - FS, 97)

    phase = estimator.get_phase_from_triggers(ALPHA, triggers)
    expected = np.angle(signal.hilbert(
        signal.sosfiltfilt(SOS_6TH_ORDER, ALPHA))[triggers], deg=True) % 360

    assert np.all(np.isfinite(phase))
    np.testing.assert_allclose(phase, expected)


def test_estimator_sos_real_time_filter_works_for_power_threshold():
    sos = signal.butter(4, [8, 12], btype='bandpass', fs=FS, output='sos')
    estimator = Estimator(sos, SOS_6TH_ORDER, FS)
    window = ALPHA[:FS]

    assert estimator.check_power_threshold(window, 0.5) is True
    assert estimator.check_power_threshold(window, 2) is False


def test_estimator_rejects_unstable_ba_filter():
    unstable = signal.butter(6, [8, 12], btype='bandpass', fs=FS)

    with pytest.raises(ValueError, match="unstable.*output='sos'"):
        Estimator(real_time_filter_fir, unstable, FS)
    with pytest.raises(ValueError, match='real_time_filter is unstable'):
        Estimator(unstable, ground_truth_filter_fir, FS)


@pytest.mark.parametrize('bad_filter', [
    np.ones((3, 5)),
    np.ones((1, 101)),
    np.array([[1.0, 0.5], [0.0, 1.0]]),
])
def test_estimator_rejects_malformed_2d_filters(bad_filter):
    with pytest.raises(ValueError):
        Estimator(real_time_filter_fir, bad_filter, FS)


@pytest.mark.parametrize('triggers,exception,message', [
    ([-5], ValueError, r'range \[0, 20480\)'),
    ([len(ALPHA)], ValueError, r'range \[0, 20480\)'),
    ([10, -1, 20], ValueError, 'first: -1'),
    ([1.5], TypeError, 'whole-number'),
    (['a'], TypeError, 'integer sample indices'),
])
def test_get_phase_from_triggers_rejects_invalid_triggers(
        triggers, exception, message):
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)

    with pytest.raises(exception, match=message):
        estimator.get_phase_from_triggers(ALPHA, triggers)
    with pytest.raises(exception, match=message):
        estimator.get_waveforms_from_triggers(ALPHA, triggers, 0.1, 0.1)


def test_get_phase_from_triggers_accepts_whole_number_floats_and_empty():
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)

    np.testing.assert_array_equal(
        estimator.get_phase_from_triggers(ALPHA, [4096.0, 5000.0]),
        estimator.get_phase_from_triggers(ALPHA, [4096, 5000]))
    assert len(estimator.get_phase_from_triggers(ALPHA, [])) == 0


def test_phase_stats_are_nan_without_triggers():
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)

    assert np.isnan(estimator.phase_accuracy_from_triggers(ALPHA, [], 0))
    assert np.isnan(estimator.mean_phase_from_triggers(ALPHA, []))
    assert np.isnan(estimator.std_phase_from_triggers(ALPHA, []))

    stats = estimator.compute_phase_stats(ALPHA, [], target_phase=0)
    assert len(stats['phase']) == 0
    assert np.isnan(stats['mean'])
    assert np.isnan(stats['accuracy'])


@pytest.mark.parametrize('tmin,tmax,message', [
    (-0.1, 0.1, "not MNE's negative tmin"),
    (0.1, -0.1, 'tmax must be non-negative'),
    (0, 0, 'no samples'),
])
def test_get_waveforms_from_triggers_rejects_negative_times(
        tmin, tmax, message):
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)

    with pytest.raises(ValueError, match=message):
        estimator.get_waveforms_from_triggers(ALPHA, [4096], tmin, tmax)


def test_phase_methods_default_to_degrees_in_full_circle():
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)
    # a sine is at -90 degrees (270) on whole seconds
    triggers = np.arange(2, 8) * FS

    phase = estimator.get_phase_from_triggers(ALPHA, triggers)
    mean = estimator.mean_phase_from_triggers(ALPHA, triggers)
    mean_rad = estimator.mean_phase_from_triggers(
        ALPHA, triggers, degree=False)

    assert np.all((phase >= 0) & (phase < 360))
    np.testing.assert_allclose(phase, 270, atol=1)
    assert mean == pytest.approx(270, abs=1)
    assert mean_rad == pytest.approx(np.deg2rad(mean))
    assert estimator.std_phase_from_triggers(ALPHA, triggers) == \
        pytest.approx(np.rad2deg(estimator.std_phase_from_triggers(
            ALPHA, triggers, degree=False)))
    assert estimator.phase_accuracy_from_triggers(
        ALPHA, triggers, 270) == pytest.approx(1, abs=0.01)


def test_compute_phase_stats_matches_individual_methods_and_filters_once(
        monkeypatch):
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)
    rng = np.random.default_rng(0)
    data = ALPHA + rng.normal(0, 0.5, len(ALPHA))
    triggers = rng.integers(FS, len(data) - FS, 50)

    calls = []
    original_filter = estimator._filter_data

    def counting_filter(dsp_filter, window):
        calls.append(len(window))
        return original_filter(dsp_filter, window)

    monkeypatch.setattr(estimator, '_filter_data', counting_filter)

    stats = estimator.compute_phase_stats(data, triggers, target_phase=90)
    np.testing.assert_allclose(
        stats['phase'], estimator.get_phase_from_triggers(data, triggers))
    assert stats['mean'] == pytest.approx(
        estimator.mean_phase_from_triggers(data, triggers))
    assert stats['std'] == pytest.approx(
        estimator.std_phase_from_triggers(data, triggers))
    assert stats['accuracy'] == pytest.approx(
        estimator.phase_accuracy_from_triggers(data, triggers, 90))
    estimator.get_waveforms_from_triggers(data, triggers, 0.1, 0.1)

    assert calls == [len(data)]
    assert estimator.compute_phase_stats(data, triggers)['accuracy'] is None


def test_ground_truth_cache_is_invalidated_by_new_data_or_filter():
    estimator = Estimator(real_time_filter_fir, ground_truth_filter_fir, FS)
    data = ALPHA.copy()
    triggers = [4096, 5000]

    before = estimator.get_phase_from_triggers(data, triggers)

    # in-place change to the same array
    data[:] = np.cos(2 * np.pi * 10 * TIME)
    after_data_change = estimator.get_phase_from_triggers(data, triggers)
    assert not np.allclose(before, after_data_change)

    estimator.ground_truth_filter = SOS_6TH_ORDER
    expected = np.angle(signal.hilbert(
        signal.sosfiltfilt(SOS_6TH_ORDER, data))[triggers], deg=True) % 360
    np.testing.assert_allclose(
        estimator.get_phase_from_triggers(data, triggers), expected)
