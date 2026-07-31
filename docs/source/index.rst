EEGPhasePy documentation
========================

EEGPhasePy is an open-source toolkit for developing real-time EEG phase estimation and analyzing the results of EEG phase estimation results. It includes implementations of common statistics and figures used to quantify EEG phase estimation results. EEGPhasePy also includes implementations of EEG phase estimation algorithms alongside optimizations of parameters of these algorithms to improve their performance on a participant-by-participant basis.

Getting started
----------------
We will walk through getting started with your first real-time EEG phase estimation experiment using EEGPhasePy, including real-time phase estimation and analysis of the outcome of the experiment. To get started, install the package from PyPi by running the command:

.. code-block:: Python
    :caption: Install EEGPhasePy

    pip install eegphasepy

Once you've installed EEGPhasePy, you will need to select the model you're planning to run phase estimation with. Below, we are using the Educated Temporal Prediction (ETP) model :cite:t:`Shirinpour2020-ef`, see :ref:`API Reference <api-estimators>` for the models currently implemented in EEGPhasePy.

.. code-block:: Python
   :caption: Import necessary packages

   from EEGPhasePy.estimators import ETP

We will perform mu-band phase estimation from the C3 electrode

Real-time phase estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a live experiment your amplifier sends packets of samples at a fixed rate. On each packet arrival you append the new samples to a rolling 2 s buffer, run ETP on that buffer, and schedule a trigger to fire at the sample offset ETP returns. ``predict`` returns the number of samples from the *end of the current window* at which the target phase is expected — converting that to wall-clock time (``offset / fs``) gives you how far in the future to schedule the stimulus.

.. code-block:: Python
   :caption: Real-time loop: rolling buffer, ETP, and trigger scheduling

   import numpy as np
   import scipy.signal as signal
   import collections, time
   from EEGPhasePy.estimators import ETP

   fs = 2000
   window_len = 2 * fs        # 2 s rolling buffer
   packet_size = int(0.06 * fs)  # samples per amplifier packet (~60 ms)

   rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
   gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)

   etp = ETP(rt_filter, gt_filter, fs)
   etp.fit(training_signal, min_ipi=int(fs * 1/12))

   buffer = collections.deque(maxlen=window_len)  # auto-drops oldest samples
   triggers = []

   # This would typically be a while loop that gets the latest packet from your amplifier, here we've just simulated a sliding window with a simulated signal
   for packet_start in range(0, len(testing_signal) - packet_size, packet_size):
       packet = testing_signal[packet_start:packet_start + packet_size]
       buffer.extend(packet)

       if len(buffer) < window_len:
           continue  # wait until the buffer is full

       window = np.array(buffer)
       try:
           offset = etp.predict(window, target_phase=0) # here we use ETP to predict the next sample of a peak on EEG
           trigger_time = time.time() + offset / fs   # absolute wall-clock time
           trigger_sample = packet_start + packet_size + offset
           triggers.append(trigger_sample)

           # schedule_trigger(trigger_time) # call your hardware trigger here
       except RuntimeError:
           pass  # no peaks found in this window

Analysing the output
~~~~~~~~~~~~~~~~~~~~~

In raw EEG, you would typically have trigger markers stored as annotations in your EEG file. EEGPhasePy just needs the sample number each trigger occurs at to analyze the outcome of your phase estimation experiment.

To quantify how well the desired phase was targetted you can use :py:meth:`~EEGPhasePy.estimators.Estimator.mean_phase_from_triggers` and :py:meth:`~EEGPhasePy.estimators.Estimator.std_phase_from_triggers`, which return the circular mean and standard deviation of the phase at each trigger (a std of 50°–70° is typical in published studies). :py:meth:`~EEGPhasePy.estimators.Estimator.phase_accuracy_from_triggers` returns a 0–1 score where 0.5 is chance, 0 is anti-phase, and 1 is perfect locking.

.. code-block:: Python
   :caption: Compute phase statistics

   mean_phase = etp.mean_phase_from_triggers(testing_signal, triggers, degree=True)
   std_phase = etp.std_phase_from_triggers(testing_signal, triggers, degree=True)
   accuracy = etp.phase_accuracy_from_triggers(testing_signal, triggers, 0)

   print(f"Mean phase: {mean_phase:.1f}°  |  Std: {std_phase:.1f}°  |  Accuracy: {accuracy * 100:.1f}%")


.. toctree::
   :caption: Guide
   :maxdepth: 3

   usage/guide
   usage/realtime


.. toctree::
   :caption: Models
   :maxdepth: 3

   usage/etp
   usage/ar


.. toctree::
   :caption: Optimizing
   :maxdepth: 3

   usage/bayesian_optimization
   usage/genetic_optimization


.. toctree::
   :caption: Analysis
   :maxdepth: 3

   usage/waveform_plots
   usage/polar_histograms
   usage/stats


.. toctree::
   :caption: Examples
   :maxdepth: 1

   gallery_examples/index


.. toctree::
   :caption: Reference
   :maxdepth: 3

   api

Bibliography
------------------
.. bibliography:: references.bib
    :style: unsrt
    :all: