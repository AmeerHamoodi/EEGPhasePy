Genetic optimization of PHASTIMATE
=====================================

Two parameters of :py:class:`EEGPhasePy.estimators.PHASTIMATE` have a large
influence on phase estimation accuracy: ``window_edge`` and ``ar_order``.
The optimal values are signal-dependent, so EEGPhasePy provides a built-in
optimizer to tune them automatically from a short resting-state recording.

What the parameters control
-----------------------------

``window_edge`` (milliseconds)
    After the real-time filter is applied to the current window, the last
    ``window_edge`` ms of the filtered output are discarded before the AR
    model extrapolates. A larger value protects against filter ringing at
    the cost of requiring a longer AR forecast, which accumulates more error.
    The optimizer searches in the range 10–80 ms (capped at window_len / 8),
    in steps of 5 ms.

``ar_order``
    The number of past samples the autoregressive model uses when
    extrapolating beyond the window edge. Higher orders can capture more
    complex spectral structure but risk over-fitting to noise. The optimizer
    searches from 1 up to 10 % of the sampling rate.

When to use optimization
--------------------------

Default parameters work reasonably well for standard mu/alpha bands (8–12 Hz)
with typical signal quality. Consider running optimization when:

- You observe high inter-individual variability in phase accuracy across
  participants.
- You are working with a frequency band that is less commonly studied, where
  default parameters may not be well matched to the spectral properties of
  the signal.
- You have resting-state data available and want to squeeze out additional
  accuracy before the main experiment.

How genetic optimization works here
--------------------------------------

Genetic optimization treats the search as an evolutionary process. An initial
population of candidate ``(window_edge, ar_order)`` pairs is evaluated against
the training data. At each generation, the fittest candidates are selected as
parents and recombined with random mutation to produce the next generation.
Over 20 generations the population converges toward high-accuracy parameter
combinations.

Compared to Bayesian optimization, the genetic approach explores the search
space more broadly and can escape local optima, but typically requires more
fitness evaluations to converge. It is a good choice when you want broad
coverage of the parameter space or when the accuracy surface is expected to be
irregular. The original PHASTIMATE paper (:cite:t:`Zrenner2020-zb`) used a
genetic search, so this method also ensures the closest methodological
alignment with the published toolbox.

Usage
------

Pass your resting-state or training EEG data and specify ``method="genetic"``:

.. code-block:: python
   :caption: Genetic optimization of PHASTIMATE

   phastimate.optimize_parameters(resting_data, method="genetic")

   # Inspect the tuned parameters
   print(f"window_edge: {phastimate.window_edge} ms")
   print(f"ar_order:    {phastimate.ar_order}")

``optimize_parameters`` updates the instance in-place — subsequent calls to
:py:meth:`~EEGPhasePy.estimators.PHASTIMATE.predict` will use the tuned values
automatically. The data passed in should be the same channel and frequency band
you intend to run phase estimation on.

.. minigallery:: ../examples/ar-genetic-*
