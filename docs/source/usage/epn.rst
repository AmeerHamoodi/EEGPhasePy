EPN-based phase estimation
===========================

Background
------------
The EEG Phase Prediction Network (EPN) is a machine learning approach to phase estimation that maps a window of raw EEG directly to
the phase at the current point in time. EPN was first described by :cite:t:`Liu2025-epn`, so please cite :cite:t:`Liu2025-epn` when using the
:py:class:`EEGPhasePy.estimators.EPN` class of EEGPhasePy.

.. important:: Citing EPN

   If you use :py:class:`EEGPhasePy.estimators.EPN` in your research, please cite the original EPN paper:

   Liu, H., Qi, Z., Wang, Y., Yang, Z., Fan, L., Zuo, N., & Jiang, T. (2025). A Novel Real-time Phase Prediction Network in EEG Rhythm.
   *Neuroscience Bulletin*, 41(3), 391-405. `doi:10.1007/s12264-024-01321-z <https://doi.org/10.1007/s12264-024-01321-z>`_

The network itself is small. It takes the backward difference of a 500 ms window of EEG (:math:`y(n) = x(n + 1) - x(n)`), which removes slow electrode
drift, and smooths it with an average pooling layer that acts as a moving average filter. A fully connected layer with 500 units and a ReLU activation
expands these features, dropout is applied during training to prevent overfitting, and a final layer maps the features to two outputs. The two outputs are
treated as the cosine and sine of the phase, so the phase is recovered with :math:`\arctan2`. Unlike ETP and AR, the window is never band-pass filtered.
Instead, the network is trained on resting-state EEG from each participant to reproduce the "ground-truth" phase, which is obtained by non-causally filtering
the full recording and applying a Hilbert transform.

Like AR, EPN estimates the phase at the current point in time rather than predicting when the target phase will occur. This means it benefits from a
high packet send rate from your EEG amplifier for the same reasons described on the :doc:`AR page <ar>`. :cite:t:`Liu2025-epn` found EPN had a higher accuracy
and a lower variance than AR, ETP and a multi-layer filter architecture on pre-recorded, simulated and real-time data in the sensorimotor :math:`\mu` rhythm,
without preselecting participants or trials. The trade-off is that a few minutes of resting-state EEG has to be collected to train the network before the
experiment, similar to ETP.

Implementation notes
~~~~~~~~~~~~~~~~~~~~~
Our implementation follows the architecture in :cite:t:`Liu2025-epn`, but there are a few differences worth knowing about:

* The network is implemented and trained in NumPy, so no deep learning framework is needed. Prediction on a single window takes tens of microseconds on a CPU.
* EEGPhasePy uses the Hilbert phase convention used by the rest of the package, where 0 degrees is a peak and 180 degrees is a trough. :cite:t:`Liu2025-epn`
  report phases with 0 degrees as the rising edge and 90 degrees as the peak.
* :cite:t:`Liu2025-epn` cut their training data into 6.16 s epochs and trained for 3000 epochs with a learning rate of :math:`10^{-5}`. Here, a training window is taken
  every ``window_step`` ms (10 ms by default) which gives far more windows per epoch, so the defaults are 50 epochs with a learning rate of :math:`10^{-3}`.
  Each of these can be changed in :py:meth:`~EEGPhasePy.estimators.EPN.fit`.
* The backward difference is divided by the standard deviation of the training data's backward difference, so the network behaves the same whether your data is in
  volts or microvolts.
* :cite:t:`Liu2025-epn` resampled their data to 500 Hz and applied a C4-centred Hjorth (surface Laplacian) spatial filter before EPN. All estimators in EEGPhasePy
  work on a single channel, so any spatial filtering or downsampling should be done before passing data in. We recommend 500 Hz to match the original work. It also
  keeps training fast since the size of the network grows with the number of samples in the window.
* The average pooling kernel is given in ms (``pool_len``) to match the rest of the package. The default of 26 ms is the 13 sample kernel used by :cite:t:`Liu2025-epn` at 500 Hz.
* ``real_time_filter`` is not used to estimate phase, since EPN works on the raw window. It is still required so that
  :py:meth:`~EEGPhasePy.estimators.Estimator.check_power_threshold` works the same way as it does for the other estimators.


Usage
---------
We'll start off by importing the EPN class, :mod:`numpy` to simulate EEG data and :mod:`scipy.signal` for filtering.

.. doctest::
   :hide:
   :pyversion: == 3.12

.. testsetup:: *

   import numpy as np
   import scipy.signal as signal

   from EEGPhasePy.estimators import EPN

.. code-block:: Python
   :caption: Imports

   import numpy as np
   import scipy.signal as signal

   from EEGPhasePy.estimators import EPN

Next, we'll create two 200 s long signals simulating the human alpha rhythm (assuming 10 Hz here) and add some gaussian noise. We use a sampling rate of 500 Hz to match
:cite:t:`Liu2025-epn`. Our training signal will be used to fit EPN and then we will test it on the testing signal.

.. testcode::

   fs = 500
   time_data = np.arange(0, 200, 1/fs)

   training_signal_clean = np.sin(2 * np.pi * 10 * time_data)
   training_signal = training_signal_clean + np.random.normal(0, 2, len(time_data))

   testing_signal_clean = np.sin(2 * np.pi * 10 * time_data)
   testing_signal = testing_signal_clean + np.random.normal(0, 2, len(time_data))

After our signals have been created, we need to construct our real-time and ground-truth filter. The ground-truth filter is used to create the phase that EPN is
trained to estimate and to compute phase stats (e.g. circular mean and std) based on triggers. EPN doesn't filter the window passed into
:py:meth:`~EEGPhasePy.estimators.EPN.predict`, so the real-time filter is only used if you check the power of the window with
:py:meth:`~EEGPhasePy.estimators.Estimator.check_power_threshold`.

.. note:: A note on the "ground truth" filter

   We use ground-truth here quite loosely. It is very challenging to
   obtain a true "ground-truth" for EEG phase. If you're interested in understanding
   more of the nuance associated with this, check out :cite:t:`Zrenner2020-zb`.
   Since EPN learns to reproduce the output of the ground-truth filter, the choice of
   this filter directly shapes what the network learns.

.. testcode::

   rt_filter = signal.firwin(64, [8, 12], fs=fs, pass_zero=False)
   gt_filter = signal.firwin(151, [8, 12], fs=fs, pass_zero=False)

Next, we will instantiate EPN and fit it to our training data. The defaults match the network described by :cite:t:`Liu2025-epn`. Passing ``random_state`` makes
training reproducible and ``verbose=True`` will print the training loss after each epoch. The loss for each epoch is also stored in ``epn.loss_history`` so you can
check that it has levelled off.

.. testcode::

   epn = EPN(rt_filter, gt_filter, fs)
   epn.fit(training_signal, random_state=0)

Then, we will run a pseudo-real-time simulation using the testing signal. Like :py:class:`~EEGPhasePy.estimators.PHASTIMATE`, :py:meth:`~EEGPhasePy.estimators.EPN.predict`
returns a boolean value about whether the current phase is your target phase, within a tolerance (5 degrees by default). The ``target_phase`` argument is specified in
**degrees** (e.g. ``0`` for a peak, ``180`` for a trough, ``270`` for a rising zero-crossing). The phase EPN estimates is the phase at the sample immediately after the
window, which is where a trigger would land, so the trigger sample is ``window_start + window_len``.

Here, a new window is passed in every 2 ms, which is the same as an amplifier sending a packet at 500 Hz. After each trigger we skip ahead by one window so that
triggers aren't delivered back-to-back.

.. testcode::

   window_start = 0
   window_len = int(0.5*fs)
   window_step = int(0.002*fs) # using a step of 2 ms

   triggers = []

   while window_start + window_len < len(testing_signal):
       window = testing_signal[window_start:window_start + window_len]

       if epn.predict(window, 0, 10):
           triggers.append(window_start + window_len)
           window_start += window_len

       window_start += window_step

   polar_hist_fig = epn.polar_histogram_from_triggers(testing_signal, triggers)
   waveform_fig = epn.plot_mean_std_waveform_from_triggers(testing_signal, triggers, tmin=0.1, tmax=0.1)

If you want the phase itself rather than a yes/no answer, :py:meth:`~EEGPhasePy.estimators.EPN.estimate_phase` returns the estimated phase in degrees (0 to 360).

.. testcode::

   current_phase = epn.estimate_phase(window)

Using EPN in real-time
~~~~~~~~~~~~~~~~~~~~~~~
The network has a fixed number of inputs, so it always uses the most recent ``window_len`` ms of the data passed in. This means you can pass a longer rolling buffer
straight into :py:meth:`~EEGPhasePy.estimators.EPN.predict` or :py:meth:`~EEGPhasePy.estimators.EPN.estimate_phase`; passing less than ``window_len`` ms of data raises a ``ValueError``.

Training takes anywhere from a few seconds to a minute depending on your sampling rate and the amount of training data. Typically, you would collect resting-state EEG at the
start of the session, fit EPN and then use the same object for the rest of the session. A fitted EPN object can also be saved with :mod:`pickle` and loaded later:

.. code-block:: Python
   :caption: Saving and loading a fitted EPN model

   import pickle

   with open("sub-01_epn.pkl", "wb") as f:
       pickle.dump(epn, f)

   with open("sub-01_epn.pkl", "rb") as f:
       epn = pickle.load(f)

Examples
----------
.. minigallery:: ../examples/epn-*
