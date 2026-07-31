Bayesian optimization of PHASTIMATE
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
    The optimizer searches in the range 5–60 ms (capped at window_len / 8).

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

How Bayesian optimization works here
--------------------------------------

Bayesian optimization builds a Gaussian process surrogate of the accuracy
surface over the ``(window_edge, ar_order)`` search space. After a small
number of random initial evaluations it uses the surrogate to decide which
point to evaluate next, balancing exploration (regions with high uncertainty)
against exploitation (regions already known to be good). This makes it
particularly sample-efficient: it reaches a near-optimal solution with far
fewer fitness evaluations than a grid search.

Each evaluation fits an AR model with the candidate parameters and scores it
against the ground-truth phase from the training data. The optimizer runs
10 random initial evaluations followed by 100 directed iterations, then sets
``window_edge`` and ``ar_order`` on the :py:class:`~EEGPhasePy.estimators.PHASTIMATE`
instance in-place.

Usage
------

Pass your resting-state or training EEG data and specify ``method="bayesian"``:

.. code-block:: python
   :caption: Bayesian optimization of PHASTIMATE

   phastimate.optimize_parameters(resting_data, method="bayesian")

   # Inspect the tuned parameters
   print(f"window_edge: {phastimate.window_edge} ms")
   print(f"ar_order:    {phastimate.ar_order}")

``optimize_parameters`` updates the instance in-place — subsequent calls to
:py:meth:`~EEGPhasePy.estimators.PHASTIMATE.predict` will use the tuned values
automatically. The data passed in should be the same channel and frequency band
you intend to run phase estimation on.

.. minigallery:: ../examples/ar-bayesian-*
