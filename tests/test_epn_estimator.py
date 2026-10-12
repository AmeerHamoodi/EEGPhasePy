import os
import pickle

import numpy as np
import pytest
import scipy.signal as signal
import scipy.stats as stats

if not os.getenv('GITHUB_ACTIONS') == 'true':
    import mne

from EEGPhasePy.estimators import EPN

FS = 500
ALPHA_FREQUENCY = 10

real_time_filter = signal.firwin(64, [8, 12], fs=FS, pass_zero=False)
ground_truth_filter = signal.firwin(151, [8, 12], fs=FS, pass_zero=False)


def simulate_alpha(duration, noise_std, phase_offset=0.0, seed=0):
    '''
    Simulate a 10 Hz alpha rhythm with additive gaussian noise

    Returns
    -------
    noisy_signal : array (n_samples,)
    true_phase : array (n_samples,)
        Hilbert phase of the clean rhythm in degrees (0 = peak)
    '''
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, 1 / FS)
    angle = 2 * np.pi * ALPHA_FREQUENCY * time + phase_offset
    noisy_signal = np.sin(angle) + rng.normal(0, noise_std, len(time))
    # the Hilbert phase of sin(x) is x - pi/2
    true_phase = np.rad2deg(angle - np.pi / 2) % 360
    return noisy_signal, true_phase


def circular_error(estimated, target):
    '''
    Signed difference between phases in degrees, wrapped to [-180, 180)
    '''
    return (np.asarray(estimated) - target + 180) % 360 - 180


def fit_epn(training_data, **fit_kwargs):
    fit_kwargs = {'epochs': 30, 'window_step': 20, 'random_state': 0,
                  **fit_kwargs}
    epn = EPN(real_time_filter, ground_truth_filter, FS)
    return epn.fit(training_data, **fit_kwargs)


@pytest.fixture(scope='module')
def training_signal():
    return simulate_alpha(60, noise_std=1, seed=1)[0]


@pytest.fixture(scope='module')
def fitted_epn(training_signal):
    return fit_epn(training_signal)


def test_epn_construct_defaults_match_paper():
    epn = EPN(real_time_filter, ground_truth_filter, FS)

    assert epn.window_len == 500
    assert epn.pool_len == 26
    assert epn.hidden_size == 500
    assert epn.dropout == 0.9
    assert epn.real_time_filter is real_time_filter
    assert epn.ground_truth_filter is ground_truth_filter
    assert epn.sampling_rate == FS


def test_epn_fitted_architecture_matches_paper(fitted_epn):
    # 500 ms at 500 Hz is 250 samples, which gives 249 differences
    assert fitted_epn._n_samples == 250
    assert fitted_epn._pool_size == 13
    assert fitted_epn._w1.shape == (249, 500)
    assert fitted_epn._b1.shape == (500,)
    assert fitted_epn._w2.shape == (500, 2)
    assert fitted_epn._b2.shape == (2,)


@pytest.mark.parametrize(('kwargs', 'error', 'message'), [
    ({'sampling_rate': 500.5}, TypeError, 'Value must be an int type'),
    ({'pool_len': 26.5}, TypeError, 'Value must be an int type'),
    ({'hidden_size': 500.0}, TypeError, 'Value must be an int type'),
    ({'dropout': '0.9'}, TypeError, 'dropout must be a real number'),
    ({'dropout': True}, TypeError, 'dropout must be a real number'),
    ({'dropout': -0.1}, ValueError, 'dropout must be non-negative'),
    ({'dropout': 1}, ValueError, 'dropout must be less than 1'),
    ({'hidden_size': 0}, ValueError, 'hidden_size must be at least 1'),
    ({'pool_len': 1}, ValueError, 'pool_len is too small'),
    ({'pool_len': 600}, ValueError, 'pool_len is too large'),
])
def test_epn_construct_rejects_invalid_parameters(kwargs, error, message):
    params = {'real_time_filter': real_time_filter,
              'ground_truth_filter': ground_truth_filter,
              'sampling_rate': FS, **kwargs}

    with pytest.raises(error, match=message):
        EPN(**params)


def test_epn_construct_accepts_custom_architecture():
    epn = EPN(real_time_filter, ground_truth_filter, FS, window_len=400,
              pool_len=10, hidden_size=64, dropout=0)
    epn.fit(simulate_alpha(30, noise_std=1)[0], epochs=2, window_step=20)

    assert epn._w1.shape == (199, 64)
    assert epn._pool_size == 5
    assert epn.dropout == 0.0


@pytest.mark.parametrize(('data', 'kwargs', 'error', 'message'), [
    (1, {}, TypeError, 'Value must be an array type'),
    (np.zeros((1000, 2)), {}, ValueError,
     'The provided array has the wrong dimension'),
    (np.full(5000, np.nan), {}, ValueError,
     'training_data must contain only finite samples'),
    (np.random.default_rng(0).normal(size=400), {}, ValueError,
     'training_data is too short: expected at least 454 samples'),
    (np.ones(5000), {}, ValueError, 'training_data must not be constant'),
    (None, {'epochs': 1.5}, TypeError, 'Value must be an int type'),
    (None, {'epochs': 0}, ValueError, 'epochs must be at least 1'),
    (None, {'batch_size': 0}, ValueError, 'batch_size must be at least 1'),
    (None, {'learning_rate': '1e-3'}, TypeError,
     'learning_rate must be a real number'),
    (None, {'learning_rate': 0}, ValueError,
     'learning_rate must be positive'),
    (None, {'window_step': 10.0}, TypeError, 'Value must be an int type'),
    (None, {'window_step': 1}, ValueError, 'window_step is too small'),
    (None, {'random_state': 1.5}, TypeError, 'Value must be an int type'),
    (None, {'verbose': 'yes'}, TypeError, 'Value must be a bool type'),
])
def test_epn_fit_rejects_invalid_input(training_signal, data, kwargs, error,
                                       message):
    epn = EPN(real_time_filter, ground_truth_filter, FS)
    data = training_signal if data is None else data

    with pytest.raises(error, match=message):
        epn.fit(data, **kwargs)


def test_epn_must_be_fit_before_use(training_signal):
    epn = EPN(real_time_filter, ground_truth_filter, FS)

    with pytest.raises(RuntimeError, match='EPN has not been fit'):
        epn.estimate_phase(training_signal[:250])
    with pytest.raises(RuntimeError, match='EPN has not been fit'):
        epn.predict(training_signal[:250], 0)


def test_epn_fit_returns_self_and_reduces_loss(training_signal):
    epn = EPN(real_time_filter, ground_truth_filter, FS)
    returned = epn.fit(training_signal, epochs=10, window_step=20,
                       random_state=0)

    assert returned is epn
    assert len(epn.loss_history) == 10
    assert epn.loss_history[-1] < epn.loss_history[0]


def test_epn_refit_starts_from_scratch(training_signal):
    epn = fit_epn(training_signal, epochs=3)
    epn.fit(training_signal, epochs=2, window_step=20, random_state=0)
    refit_phase = epn.estimate_phase(training_signal[:250])

    fresh_phase = fit_epn(training_signal, epochs=2).estimate_phase(
        training_signal[:250])

    assert len(epn.loss_history) == 2
    assert refit_phase == pytest.approx(fresh_phase)


def test_epn_fit_is_reproducible_with_random_state(training_signal):
    window = training_signal[1000:1250]

    first = fit_epn(training_signal, epochs=3)
    second = fit_epn(training_signal, epochs=3)
    other_seed = fit_epn(training_signal, epochs=3, random_state=1)

    np.testing.assert_array_equal(first._w1, second._w1)
    assert first.loss_history == second.loss_history
    assert first.estimate_phase(window) == second.estimate_phase(window)
    assert not np.array_equal(first._w1, other_seed._w1)


def test_epn_fit_verbose_prints_loss(training_signal, capsys):
    fit_epn(training_signal, epochs=2, verbose=True)

    output = capsys.readouterr().out
    assert '[EPN] Epoch 1/2 - loss:' in output
    assert '[EPN] Epoch 2/2 - loss:' in output


def test_epn_fit_accepts_iir_ground_truth_filter(training_signal):
    iir_filter = np.array(signal.butter(3, [8, 12], btype='bandpass', fs=FS))
    epn = EPN(real_time_filter, iir_filter, FS)
    epn.fit(training_signal, epochs=2, window_step=20, random_state=0)

    assert 0 <= epn.estimate_phase(training_signal[:250]) < 360


def test_epn_estimate_phase_returns_degrees(fitted_epn, training_signal):
    phase = fitted_epn.estimate_phase(training_signal[:250])

    assert isinstance(phase, float)
    assert 0 <= phase < 360


def test_epn_estimate_phase_uses_most_recent_window(fitted_epn,
                                                    training_signal):
    buffer = training_signal[:1000]

    assert fitted_epn.estimate_phase(buffer) == \
        fitted_epn.estimate_phase(buffer[-250:])
    assert fitted_epn.estimate_phase(list(buffer)) == \
        fitted_epn.estimate_phase(buffer)


@pytest.mark.parametrize(('data', 'error', 'message'), [
    (5, TypeError, 'Value must be an array type'),
    (np.zeros((250, 2)), ValueError,
     'The provided array has the wrong dimension'),
    (np.full(250, np.inf), ValueError,
     'data must contain only finite samples'),
    (np.zeros(249), ValueError,
     'data window is too short: expected at least 250 samples, got 249'),
])
def test_epn_estimate_phase_rejects_invalid_data(fitted_epn, data, error,
                                                 message):
    with pytest.raises(error, match=message):
        fitted_epn.estimate_phase(data)


def test_epn_estimates_phase_at_sample_after_window():
    # With a clean rhythm the network should recover the analytic phase
    # almost exactly, so a one sample misalignment (7.2 degrees at 10 Hz and
    # 500 Hz) would show up as a bias
    epn = fit_epn(simulate_alpha(60, noise_std=0)[0])
    test_signal, true_phase = simulate_alpha(10, noise_std=0,
                                             phase_offset=1.0)

    window_ends = np.arange(250, len(test_signal) - 1, 7)
    estimated = [epn.estimate_phase(test_signal[:end]) for end in window_ends]
    errors = circular_error(estimated, true_phase[window_ends])

    mean_error = np.rad2deg(stats.circmean(np.deg2rad(errors),
                                           low=-np.pi, high=np.pi))
    assert abs(mean_error) < 2
    assert np.mean(np.abs(errors)) < 4


def test_epn_estimates_phase_of_noisy_alpha(fitted_epn):
    test_signal, true_phase = simulate_alpha(30, noise_std=1,
                                             phase_offset=0.5, seed=2)

    window_ends = np.arange(250, len(test_signal) - 1, 13)
    estimated = [fitted_epn.estimate_phase(test_signal[:end])
                 for end in window_ends]
    errors = circular_error(estimated, true_phase[window_ends])

    mean_error = np.rad2deg(stats.circmean(np.deg2rad(errors),
                                           low=-np.pi, high=np.pi))
    accuracy = 1 - np.mean(np.abs(errors)) / 180
    assert abs(mean_error) < 5
    assert accuracy > 0.9


@pytest.mark.parametrize('target_phase', [0, 90, 180, 270])
def test_epn_predict_triggers_at_target_phase(fitted_epn, target_phase):
    test_signal = simulate_alpha(30, noise_std=1, seed=3)[0]
    window_len = 250
    window_step = 1

    triggers = []
    window_start = 0
    while window_start + window_len < len(test_signal):
        window = test_signal[window_start:window_start + window_len]
        if fitted_epn.predict(window, target_phase, tolerance=10):
            triggers.append(window_start + window_len)
            window_start += window_len
        window_start += window_step

    mean_phase = fitted_epn.mean_phase_from_triggers(
        test_signal, triggers, degree=True)
    accuracy = fitted_epn.phase_accuracy_from_triggers(
        test_signal, triggers, target_phase)

    assert len(triggers) > 50
    assert abs(circular_error(mean_phase, target_phase)) < 10
    assert accuracy > 0.85


def test_epn_predict_returns_bool(fitted_epn, training_signal):
    assert isinstance(fitted_epn.predict(training_signal[:250], 0), bool)


@pytest.mark.parametrize(('phase', 'target_phase', 'tolerance', 'expected'), [
    (100.0, 100, 5, True),
    (104.0, 100, 5, True),
    (106.0, 100, 5, False),
    (358.0, 2, 5, True),
    (2.0, 358, 5, True),
    (358.0, 10, 5, False),
    (358.0, 362, 5, True),
    (358.0, -2, 0, True),
    (180.0, 0, 179.5, False),
    (180.0, 0, 180, True),
])
def test_epn_predict_compares_phase_circularly(monkeypatch, phase,
                                               target_phase, tolerance,
                                               expected):
    epn = EPN(real_time_filter, ground_truth_filter, FS)
    monkeypatch.setattr(epn, 'estimate_phase', lambda data: phase)

    assert epn.predict(np.zeros(250), target_phase, tolerance) is expected


@pytest.mark.parametrize(('args', 'error', 'message'), [
    (('0',), TypeError, 'target_phase must be a real number'),
    ((np.nan,), ValueError, 'target_phase must be finite'),
    ((0, '5'), TypeError, 'tolerance must be a real number'),
    ((0, -1), ValueError, 'tolerance must be non-negative'),
    ((0, np.inf), ValueError, 'tolerance must be finite'),
])
def test_epn_predict_rejects_invalid_arguments(fitted_epn, training_signal,
                                               args, error, message):
    with pytest.raises(error, match=message):
        fitted_epn.predict(training_signal[:250], *args)


def test_epn_survives_pickling(fitted_epn, training_signal):
    restored = pickle.loads(pickle.dumps(fitted_epn))
    window = training_signal[2000:2250]

    assert restored.estimate_phase(window) == \
        fitted_epn.estimate_phase(window)


def test_epn_inherits_power_threshold(fitted_epn):
    time = np.arange(0, 0.5, 1 / FS)
    strong_alpha = 2 * np.sin(2 * np.pi * ALPHA_FREQUENCY * time)

    assert fitted_epn.check_power_threshold(strong_alpha, 1.0) is True
    assert fitted_epn.check_power_threshold(0.1 * strong_alpha, 1.0) is False


def test_epn_real_data():
    if not os.getenv('GITHUB_ACTIONS') == 'true':
        training_data = mne.io.read_raw_curry(
            "./tests/test_data/training-rsEEG.cdt")
        testing_data = mne.io.read_raw_curry(
            "./tests/test_data/testing-rsEEG.cdt")
        fs = 2048

        c3_train_data = training_data.pick("C3").get_data()[0]
        c3_test_data = testing_data.pick("C3").get_data()[0]

        epn = EPN(signal.firwin(128, [8, 13], fs=fs, pass_zero=False),
                  signal.firwin(300, [8, 13], fs=fs, pass_zero=False), fs)
        epn.fit(c3_train_data, epochs=10, window_step=50, random_state=0)

        window_i = 0
        window_step = int(0.002 * fs)
        window_len = int(0.5 * fs)

        triggers = []

        while window_i + window_len < len(c3_test_data):
            window_data = c3_test_data[window_i:window_i + window_len]

            if epn.predict(window_data, 0, 10):
                triggers.append(window_i + window_len)
                window_i += window_len

            window_i += window_step

        mean_phase = epn.mean_phase_from_triggers(
            c3_test_data, triggers, degree=True)
        assert abs(circular_error(mean_phase, 0)) < 5
