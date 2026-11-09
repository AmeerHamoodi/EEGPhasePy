Phase estimation statistics
====================================

When performing phase estimation, you will likely need to quantify one or more of mean, standard deviation and accuracy. All estimators 
contain helper methods for automatically computing mean, standard deviation and accuracy of triggers to a target phase.

Phase units
------------
All of EEGPhasePy uses one phase convention: **degrees in the range [0, 360), where 0 is a peak and 180 is a trough**. Target phases
(``predict``, :py:meth:`~EEGPhasePy.estimators.Estimator.phase_accuracy_from_triggers` and :py:meth:`~EEGPhasePy.estimators.Estimator.compute_phase_stats`)
are always given in degrees, and every method that returns a phase returns degrees in [0, 360) by default. Methods with a ``degree`` argument
return radians in [0, :math:`2\pi`) when it is ``False``. :py:meth:`~EEGPhasePy.estimators.Estimator.get_phase_from_triggers` can also return
the half-circle range (-180 to 180) with ``full_circle=False``.

All of the methods below take the **raw (unfiltered)** EEG and apply the estimator's ``ground_truth_filter`` themselves, so do not filter the data
beforehand. Triggers must be whole-number sample indices in the range [0, n_samples); negative or out-of-range triggers raise an error instead of
wrapping around to the end of the recording.

Mean and standard deviation 
-----------------------------
Mean and standard deviation (std) are computed using :py:mod:`scipy.stats` circular statistics. To compute the mean and standard deviation of phase 
given a list of trigger samples, you can use the :py:meth:`EEGPhasePy.estimators.Estimator.mean_phase_from_triggers` and :py:meth:`EEGPhasePy.estimators.Estimator.std_phase_from_triggers`. You can specify radian or degree output
when calling each method. See below:

.. code-block:: Python
   :caption: Computing mean and std using etp

   phase_standard_deviation = etp.std_phase_from_triggers(testing_signal, triggers, degree=True)
   phase_mean = etp.mean_phase_from_triggers(testing_signal, triggers, degree=True)


Accuracy 
---------
Accuracy is defined as 50% being completely random, 0% being completely opposite to the target phase and 100% being exactly at the target phase. 
We calculate accuracy based on the equation described by :cite:t:`Shirinpour2020-ef`. You can specify the target phase when you call the :py:meth:`EEGPhasePy.estimators.Estimator.phase_accuracy_from_triggers`
method:

.. code-block:: Python
   :caption: Computing accuracy to target phase from triggers

   accuracy = etp.phase_accuracy_from_triggers(testing_signal, triggers, 0) # in this case the target phase was 0 deg

If there are no triggers, accuracy is undefined and ``nan`` is returned (as are the mean and standard deviation). An accuracy of 0 always means the
triggers were anti-phase, never that the run failed.

Computing everything at once
-----------------------------
:py:meth:`~EEGPhasePy.estimators.Estimator.compute_phase_stats` returns the phase at each trigger along with the mean, standard deviation and
accuracy, filtering the recording once:

.. code-block:: Python
   :caption: Computing all phase statistics in one call

   phase_stats = etp.compute_phase_stats(testing_signal, triggers, target_phase=0)
   print(phase_stats['mean'], phase_stats['std'], phase_stats['accuracy'])

The ground-truth filtered recording is also cached between calls, so calling the individual methods (and the plotting helpers) on the same
recording only filters it once.


Examples 
----------
.. minigallery:: ../examples/polar-histogram.py