Power threshold estimation
==========================

In phase-triggered experiments, phase estimates from low-amplitude windows can
be dominated by noise. A power threshold provides a simple gate: estimate the
power in the target frequency band and only run phase estimation or schedule a
stimulus when the window meets a chosen minimum. This helps reject weak
windows, but it does not by itself prove that a signal is a neural oscillation.

Checking a window
-----------------

Every estimator inherits
:py:meth:`~EEGPhasePy.estimators.Estimator.check_power_threshold`. Pass the
unfiltered EEG window and a minimum power threshold. The method applies the
estimator's ``real_time_filter``, computes the analytic signal with a Hilbert
transform, and takes its amplitude envelope. It removes the configured
``window_edge`` samples from both ends, squares the remaining envelope, and
compares its mean with the threshold. The threshold is inclusive: a window
passes when its power is greater than or equal to the threshold.

This is mean squared Hilbert-envelope amplitude, not a power spectral density.
Input EEG samples must be in volts; the method converts the envelope to
microvolts before calculating power. Both the returned power and the
``power_threshold`` argument are therefore in microvolts squared. The result
depends on the selected filter and window duration, so choose the threshold
using representative data collected with the same settings.

To inspect the measured power without comparing it to a cutoff, omit
``power_threshold`` and pass ``return_power=True``. The method then returns a
float in microvolts squared. In its default mode, ``power_threshold`` is
required in microvolts squared and the method returns a boolean.

.. code-block:: python

   import numpy as np
   import scipy.signal as signal
   from EEGPhasePy.estimators import ETP

   fs = 2000
   rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
   gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)
   etp = ETP(rt_filter, gt_filter, fs)

   # Set this from representative data in microvolts squared.
   minimum_power = 4.0
   window = np.asarray(current_eeg_window)

   # Assume etp has already been fitted on representative training data.
   if etp.check_power_threshold(window, minimum_power):
       samples_to_wait = etp.predict(window, target_phase=0)
       # Schedule the trigger after samples_to_wait samples.

To inspect powers and derive a cutoff from representative windows, call:

.. code-block:: python

   training_powers = [
       etp.check_power_threshold(window, return_power=True)
       for window in training_windows
   ]
   minimum_power = np.quantile(training_powers, 0.25)

Choosing a threshold
--------------------

There is no universal threshold for EEG power. Establish one from pilot or
baseline recordings, using the same electrode, reference, filter, sampling
rate, window duration, and signal units as the experiment. Inspect the power
distribution across clean oscillatory and low-quality periods, then choose a
cutoff that rejects unsuitable windows without discarding an unacceptable
proportion of usable data. Reassess the cutoff if acquisition settings or
preprocessing change.

When comparing against a cutoff, ``power_threshold`` must be a finite,
non-negative real number in microvolts squared. The input window must contain
finite real numeric samples in volts and be long enough for the filter's
padding and the configured edge removal. Invalid inputs raise an informative
``TypeError`` or ``ValueError``. Neither mode alters the estimator's phase
prediction state.

Practical guide
---------------

The gallery example below walks through estimating a cutoff from training
windows in a real resting-state EEG recording, applying it to held-out windows,
and comparing phase-estimation results with and without the power gate. Its
cutoff is an illustrative percentile, not a universal value.

.. minigallery:: ../examples/power-threshold-*
