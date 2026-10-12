import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndimage
import scipy.signal as signal
from typing import Optional, Union

from .estimator import Estimator
from ..utils.check import (
    _check_array_dimensions,
    _check_filter_coefficients,
    _check_real_array,
    _check_real_number,
    _check_type,
)


class EPN(Estimator):
    '''
    The EEG Phase Prediction Network (EPN) :cite:t:`Liu2025-epn`.
    If you use this class please cite :cite:t:`Liu2025-epn`

    EPN is a small neural network that maps a window of raw EEG directly to
    the phase at the end of the window. A more in depth description can be
    found in :cite:t:`Liu2025-epn`.

    Briefly, the network takes the backward difference of the window, smooths
    it with an average pooling layer (a moving average), expands it with a
    fully connected layer (ReLU activation, followed by dropout during
    training) and maps it to two outputs. The two outputs are the cosine and
    sine of the phase, so the phase is recovered with ``arctan2``.

    Unlike ETP and PHASTIMATE, EPN does not band-pass filter the window
    before estimating phase. The network learns the mapping from raw data to
    the "ground-truth" phase, so it has to be fit on resting-state data from
    each participant before it is used.

    This implementation is written in NumPy and does not require a deep
    learning framework.
    '''

    def __init__(self,
                 real_time_filter: npt.ArrayLike,
                 ground_truth_filter: npt.ArrayLike,
                 sampling_rate: int,
                 window_len: int = 500,
                 window_edge: int = 40,
                 pool_len: int = 26,
                 hidden_size: int = 500,
                 dropout: Union[float, int] = 0.9):
        '''
        Construct a model for the EEG Phase Prediction Network (EPN)
        :cite:t:`Liu2025-epn`

        The defaults match the network described by :cite:t:`Liu2025-epn`:
        a 500 ms input window, an average pooling kernel of 13 samples at
        500 Hz (26 ms), 500 hidden units and a dropout probability of 0.9.

        Parameters
        ----------
        real_time_filter
            Filter parameters for the target frequency band. EPN does not
            filter the window before estimating phase, so this filter is only
            used by
            :py:meth:`~EEGPhasePy.estimators.Estimator.check_power_threshold`.
            Accounts for FIR or IIR filters
        ground_truth_filter
            Filter parameters for identifying "true" phase. Used to create the
            training targets when fitting EPN. See :cite:t:`Zrenner2020-zb`
            for an in-depth discussion on "true" phase. Accounts for FIR or
            IIR filters
        sampling_rate : int
            Original sampling rate of data.
        window_len : 500 | int
            Window length in ms. This sets the number of samples the network
            takes as input, so it should match the window used in real-time
        window_edge : 40 | int
            Window edge to remove in ms. EPN does not use this for phase
            estimation; it is only used by
            :py:meth:`~EEGPhasePy.estimators.Estimator.check_power_threshold`
        pool_len : 26 | int
            Length of the average pooling kernel in ms. 26 ms corresponds to
            the 13 sample kernel used by :cite:t:`Liu2025-epn` at 500 Hz
        hidden_size : 500 | int
            Number of units in the hidden fully connected layer
        dropout : 0.9 | float
            Probability of dropping each hidden unit during training. Must be
            in the range [0, 1). Dropout is not applied when estimating phase
        '''
        super().__init__(real_time_filter, ground_truth_filter,
                         sampling_rate, window_len, window_edge)

        _check_type(pool_len, ['int'])
        _check_type(hidden_size, ['int'])
        dropout = _check_real_number(dropout, 'dropout', non_negative=True)

        n_samples = int((window_len / 1000) * sampling_rate)
        pool_size = int((pool_len / 1000) * sampling_rate)

        if pool_size < 1:
            raise ValueError(
                "pool_len is too small for the given sampling rate. " +
                "Please increase pool_len")
        # the network sees n_samples - 1 differences
        if pool_size > n_samples - 1:
            raise ValueError(
                "pool_len is too large for the given window_len. " +
                "Please decrease pool_len or increase window_len")
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if dropout >= 1:
            raise ValueError("dropout must be less than 1")

        self.pool_len: int = pool_len
        self.hidden_size: int = hidden_size
        self.dropout: float = dropout
        self.loss_history: list = []

        # set by `fit`
        self._n_samples = None
        self._pool_size = None
        self._input_scale = None
        self._w1 = None
        self._b1 = None
        self._w2 = None
        self._b2 = None

    def _features(self, windows: np.ndarray) -> np.ndarray:
        '''
        Turn raw EEG windows into the network's input. Each window is
        differenced, scaled by the spread of the training data and smoothed
        by the average pooling layer.

        Parameters
        ----------
        windows : array (n_windows, n_samples)
            Raw EEG windows

        Returns
        -------
        features : array (n_windows, n_samples - 1)
            Pooled backward difference of each window
        '''
        differences = np.diff(windows, axis=-1) / self._input_scale
        # moving average, padding by repeating the edge values so the output
        # has the same length as the input
        return ndimage.uniform_filter1d(
            differences, self._pool_size, axis=-1, mode='nearest')

    def _forward(self, features: np.ndarray) -> np.ndarray:
        '''
        Run the network on pooled features without dropout

        Parameters
        ----------
        features : array (n_windows, n_samples - 1)
            Output of `_features`

        Returns
        -------
        output : array (n_windows, 2)
            The estimated cosine and sine of the phase for each window
        '''
        hidden = np.maximum(features @ self._w1 + self._b1, 0)
        return hidden @ self._w2 + self._b2

    def fit(self,
            training_data: npt.ArrayLike,
            epochs: int = 50,
            batch_size: int = 512,
            learning_rate: Union[float, int] = 1e-3,
            window_step: int = 10,
            random_state: Optional[int] = None,
            verbose: bool = False):
        '''
        Train the network on resting-state EEG and update self with the
        trained weights. Calling `fit` again retrains the network from
        scratch.

        A training window of ``window_len`` ms is taken every
        ``window_step`` ms. The target for each window is the ground-truth
        phase at the sample immediately after the window. Ground-truth phase
        is found by applying ``ground_truth_filter`` forwards and backwards
        over the whole recording, then taking the Hilbert transform. Targets
        within one ground-truth filter length of either end of the recording
        are skipped, since they are distorted by filter edge effects.

        As in :cite:t:`Liu2025-epn`, the network is trained with Adam to
        minimize the mean squared error between its outputs and the cosine
        and sine of the target phase. :cite:t:`Liu2025-epn` trained for 3000
        epochs with a learning rate of 1e-5 on a few hundred windows per
        participant. Windows overlap much more here, so each epoch contains
        far more windows, and the defaults use fewer epochs and a larger
        learning rate.

        Parameters
        ----------
        training_data : array (n_samples,)
            Unfiltered resting-state EEG from the target channel. A few
            minutes of data is recommended; :cite:t:`Liu2025-epn` trained on
            4 minutes
        epochs : 50 | int
            Number of passes over the training windows
        batch_size : 512 | int
            Number of windows per gradient update
        learning_rate : 1e-3 | float
            Learning rate for the Adam optimizer
        window_step : 10 | int
            Time in ms between the start of consecutive training windows
        random_state : None | int
            Seed used for weight initialization, shuffling and dropout. Pass
            an int to make training reproducible
        verbose : False | bool
            Whether to print the training loss after each epoch

        Returns
        -------
        self
        '''
        _check_type(training_data, ['array'])
        _check_type(epochs, ['int'])
        _check_type(batch_size, ['int'])
        _check_type(window_step, ['int'])
        _check_type(verbose, ['bool'])
        if random_state is not None:
            _check_type(random_state, ['int'])
        learning_rate = _check_real_number(learning_rate, 'learning_rate')

        training_data = _check_real_array(
            training_data, 'training_data', contents='samples')
        _check_array_dimensions(training_data, [(1,)])

        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        fs = self.sampling_rate
        n_samples = int((self.window_len / 1000) * fs)
        step = int((window_step / 1000) * fs)
        if step < 1:
            raise ValueError(
                "window_step is too small for the given sampling rate. " +
                "Please increase window_step")

        _, numerator_length, denominator_length = \
            _check_filter_coefficients(
                self.ground_truth_filter, 'ground_truth_filter')
        margin = max(numerator_length, denominator_length)

        # need at least one window plus a filter length on either side of
        # its target, and filtfilt pads by 3 filter lengths
        minimum_length = max(max(n_samples, margin) + margin + 1,
                             3 * margin + 1)
        if len(training_data) < minimum_length:
            raise ValueError(
                "training_data is too short: expected at least "
                f"{minimum_length} samples for the window length and "
                f"ground-truth filter, got {len(training_data)}")

        # index of the sample whose phase each window is trained to estimate.
        # The window itself is the n_samples samples before it
        target_samples = np.arange(
            max(n_samples, margin), len(training_data) - margin, step)

        # ground truth is hard to define for phase estimation, see Zrenner
        # et al., 2020 for a more detailed discussion
        filtered_data = self._filter_data(
            self.ground_truth_filter, training_data)
        ground_truth_phase = np.angle(signal.hilbert(filtered_data))

        input_scale = np.std(np.diff(training_data))
        if input_scale == 0:
            raise ValueError("training_data must not be constant")

        self._n_samples = n_samples
        self._pool_size = int((self.pool_len / 1000) * fs)
        self._input_scale = input_scale

        # same initialization as torch.nn.Linear, which the original EPN
        # was built with
        rng = np.random.default_rng(random_state)
        n_inputs = n_samples - 1
        bound = 1 / np.sqrt(n_inputs)
        self._w1 = rng.uniform(-bound, bound, (n_inputs, self.hidden_size))
        self._b1 = rng.uniform(-bound, bound, self.hidden_size)
        bound = 1 / np.sqrt(self.hidden_size)
        self._w2 = rng.uniform(-bound, bound, (self.hidden_size, 2))
        self._b2 = rng.uniform(-bound, bound, 2)

        targets = np.column_stack([
            np.cos(ground_truth_phase[target_samples]),
            np.sin(ground_truth_phase[target_samples])])
        window_offsets = np.arange(-n_samples, 0)

        # Adam state
        params = [self._w1, self._b1, self._w2, self._b2]
        first_moments = [np.zeros_like(param) for param in params]
        second_moments = [np.zeros_like(param) for param in params]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        n_updates = 0

        keep_probability = 1 - self.dropout
        self.loss_history = []

        for epoch in range(epochs):
            order = rng.permutation(len(target_samples))
            squared_error = 0

            for batch_start in range(0, len(order), batch_size):
                batch = order[batch_start:batch_start + batch_size]
                windows = training_data[
                    target_samples[batch, None] + window_offsets]
                features = self._features(windows)

                # forward pass with inverted dropout on the hidden layer
                pre_activation = features @ self._w1 + self._b1
                dropout_mask = (rng.random(pre_activation.shape) <
                                keep_probability) / keep_probability
                hidden = np.maximum(pre_activation, 0) * dropout_mask
                output = hidden @ self._w2 + self._b2

                error = output - targets[batch]
                squared_error += np.sum(error ** 2)

                # backward pass for the mean squared error
                grad_output = 2 * error / error.size
                grad_pre_activation = (grad_output @ self._w2.T) * \
                    dropout_mask * (pre_activation > 0)
                grads = [features.T @ grad_pre_activation,
                         grad_pre_activation.sum(axis=0),
                         hidden.T @ grad_output,
                         grad_output.sum(axis=0)]

                n_updates += 1
                for param, grad, first, second in zip(
                        params, grads, first_moments, second_moments):
                    first *= beta1
                    first += (1 - beta1) * grad
                    second *= beta2
                    second += (1 - beta2) * grad ** 2
                    first_corrected = first / (1 - beta1 ** n_updates)
                    second_corrected = second / (1 - beta2 ** n_updates)
                    param -= learning_rate * first_corrected / \
                        (np.sqrt(second_corrected) + epsilon)

            self.loss_history.append(squared_error / targets.size)

            if verbose:
                print(f"[EPN] Epoch {epoch + 1}/{epochs} - loss: "
                      f"{self.loss_history[-1]:.4f}")

        return self

    def estimate_phase(self, data: npt.ArrayLike) -> float:
        '''
        Estimate the phase at the end of the current window

        Only the most recent ``window_len`` ms of ``data`` are used, so a
        longer rolling buffer can be passed in directly.

        Parameters
        ----------
        data : array_like (n_samples)
            Window of unfiltered EEG data from the target channel. Must
            contain at least ``window_len`` ms of data

        Returns
        -------
        phase : float
            Estimated phase in degrees (0 to 360) at the sample immediately
            after the window. 0 is a peak and 180 is a trough
        '''
        if self._w1 is None:
            raise RuntimeError(
                "EPN has not been fit. Call `fit` with resting-state data " +
                "before estimating phase")

        _check_type(data, ['array'])
        window = _check_real_array(data, 'data', contents='samples')
        _check_array_dimensions(window, [(1,)])

        if len(window) < self._n_samples:
            raise ValueError(
                f"data window is too short: expected at least "
                f"{self._n_samples} samples, got {len(window)}")

        features = self._features(window[None, -self._n_samples:])
        cos_phase, sin_phase = self._forward(features)[0]

        return float(np.rad2deg(np.arctan2(sin_phase, cos_phase)) % 360)

    def predict(self,
                data: npt.ArrayLike,
                target_phase: Union[float, int],
                tolerance: Union[float, int] = 5) -> bool:
        '''
        Predict whether the phase at the current time matches the target phase

        Parameters
        -----------
        data : array_like (n_samples)
            Window of unfiltered EEG data from the target channel. Must
            contain at least ``window_len`` ms of data
        target_phase : float | int
            The phase in degrees that the current phase should match
        tolerance : float | int
            The tolerance between the current phase and target phase. The
            current phase has to be within `tolerance` degrees of the target
            phase for this method to return `True`

        Returns
        --------
        current_phase_matches_target : bool
            Whether the current phase matches the target phase within the
            given tolerance
        '''
        target_phase = _check_real_number(target_phase, 'target_phase')
        tolerance = _check_real_number(
            tolerance, 'tolerance', non_negative=True)

        phase = self.estimate_phase(data)

        # smallest angle between the two phases, handling the wrap at 360
        difference = abs((phase - target_phase + 180) % 360 - 180)

        return bool(difference <= tolerance)
