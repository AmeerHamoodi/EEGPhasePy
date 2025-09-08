Phase estimation statistics
====================================

When performing phase estimation, you will likely need to quantify one or more of mean, standard deviation and accuracy. All estimators 
contain helper methods for automatically computing mean, standard deviation and accuracy of triggers to a target phase.

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
Accuracy is defined as 50% begin completely random, 0% begin completely opposite to the target phase and 100% being exactly at the target phase. 
We calculate accuracy based no the equation described by :cite:t:`Shirinpour2020-ef`. You can specify the target phase when you call the :py:meth:`EEGPhasePy.estimators.Estimator.phase_accuracy_from_triggers`
method:

.. code-block:: Python
   :caption: Computing accuracy to target phase from triggers

   accuracy = etp.phase_accuracy_from_triggers(testing_signal, triggers, 0)


Examples 
----------

.. nbgallery::

   ../examples/etp-based-phase-estimation
   ../examples/ar-phase-estimation