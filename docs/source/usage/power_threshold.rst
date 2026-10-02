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
Its units are the square of the input signal's amplitude units (for example,
microvolts squared when the input is measured in microvolts). The result
depends on the selected filter, window duration, and acquisition scale, so
choose the threshold using representative data collected with the same
settings.

.. code-block:: python

   import numpy as np
   import scipy.signal as signal
   from EEGPhasePy.estimators import ETP

   fs = 2000
   rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
   gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)
   etp = ETP(rt_filter, gt_filter, fs)

   # Set this from representative data in the signal's squared amplitude units.
   minimum_power = 4.0
   window = np.asarray(current_eeg_window)

    # Assume etp has already been fitted on representative training data.
   if etp.check_power_threshold(window, minimum_power):
       samples_to_wait = etp.predict(window, target_phase=0)
       # Schedule the trigger after samples_to_wait samples.

Choosing a threshold
--------------------

There is no universal threshold for EEG power. Establish one from pilot or
baseline recordings, using the same electrode, reference, filter, sampling
rate, window duration, and signal units as the experiment. Inspect the power
distribution across clean oscillatory and low-quality periods, then choose a
cutoff that rejects unsuitable windows without discarding an unacceptable
proportion of usable data. Reassess the cutoff if acquisition settings or
preprocessing change.

``power_threshold`` must be a finite, non-negative real number. The input
window must contain finite real numeric samples and be long enough for the
filter's padding and the configured edge removal. Invalid inputs raise an
informative ``TypeError`` or ``ValueError``. The method returns a boolean and
does not alter the estimator's phase prediction state.