Bayesian optimization of PHASTIMATE
=====================================

The :py:meth:`EEGPhasePy.estimators.PHASTIMATE` parameters of ``window_edge`` and ``ar_order`` significantly impact the performance of :py:meth:`EEGPhasePy.estimators.PHASTIMATE` performance.
In certain cases, such as when dealing with high inter-individual variability or working with a frequency band that is not commonly studied, using
an optimization algorithm to select these parameters is ideal.

.. minigallery:: ../examples/ar-bayesian-*
