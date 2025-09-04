ETP-based phase estimation
===========================

Background
------------
Educated Temporal Prediction (ETP) is a form of phase estimation that uses the time of the latest peak in the current time window and the average
inter-peak-interval to predict the next time the desired phase will occur at. ETP was first described by :cite:t:`Shirinpour2020-ef`, so please cite
:cite:t:`Shirinpour2020-ef` when using the ETP class of EEGPhasePy.

The primary benefit of ETP over other phase estimation methods is its ability to work in environments with large and uncertain (within limits) 
delays. This is especially useful considering most EEG amplifiers don't make gaurentees about communication delays and the computer running ETP in real-time will likely suffer from
jitter in scheduling your phase triggered stimulus (e.g. tES or TMS). ETP allows you to account for these delays while suffering less of an accuracy
dip compared to other phase estimation algorithms


Usage
---------
Using ETP is quite simple. We'll start off by importing the ETP class, numpy to simulate EEG data and scipy.signal for filtering.

.. code-block:: Python
   :caption: Imports
   import numpy as np
   import scipy.signal as signal

   from EEGPhasePy.estimators import ETP

Next, we'll create 2 200s long signals simulating the human alpha rhythym (assuming 10hz here) and adding in some gaussian noise.
Our training signal will be used to fit the ETP algorithm and then we will test it on the testing signal.

.. code-block:: Python
   :caption: Simulating data

   fs = 2000
   time_data = np.arange(0, 200, 1/fs)

   training_signal_clean = np.sin(2 * np.pi * 10 * time_data)
   training_signal = training_signal_clean + np.random.normal(0, 2, len(time_data))

   testing_signal_clean = np.sin(2 * np.pi * 10 * time_data)
   testing_signal = testing_signal_clean + np.random.normal(0, 2, len(time_data))

After our signals have been created, we need to construct our real-time and ground-truth filter. For simplicity, 
we have assumed that a higher order filter will allow us to effectively obtain ground truth data for most of the signal. 
The purpose of the ground-truth filter is to obtain the true phase. This filter would be used when training ETP and for 
computing phase stats (e.g. circular mean and std) based on triggers. The real-time filter is used to filter the raw window
of EEG data passed into the `predict` method.

.. note:: A note on the "ground truth" filter 

   We use ground-truth here quite loosely. It is very challenging to
   obtain a true "ground-truth" for EEG phase. If you're interested in understanding
   more of the nuance associated with this, check out :cite:t:`Zrenner2020-zb`.

.. code-block:: Python
   :caption: Constructing our filters
   
   rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
   gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)

Next, we will instantiate ETP and fit it to our training data. The `min_ipi` parameter specifies the shortest
time between peaks the algorithm should expect. This is used to limit the effect phase-slips or phase-resets have
on fitting ETP.

.. code-block:: Python
   :caption: Constructing and fitting ETP
   
   etp = ETP(rt_filter, gt_filter, fs)
   etp.fit(training_signal, min_ipi=int(fs * 1/12))

Then, we will run a psuedo-real-time simulation using the testing signal. Here, we use the `predict` method of the `etp`
object. `predict` works by returning the next sample your target phase will occur at.

.. code-block:: Python
   :caption: Psuedo-real-time simulation with ETP

   window_i = 0
   window_len = int(0.5*fs)
   window_step = int(0.06*fs) # using a step of 60 ms

   triggers = []

   while window_i + window_len < len(testing_signal):
       window = testing_signal[window_i:window_i + window_len]
       
       if window_i + window_len + etp.predict(window, 0) < len(testing_signal) - window_len:
           triggers.append(window_i + window_len + etp.predict(window, 0))
       window_i += window_step
    
   polar_hist_fig = etp.polar_histogram_from_triggers(testing_signal, triggers)
   waveform_fig = etp.plot_mean_std_waveform_from_triggers(testing_signal, triggers, tmin=0.1, tmax=0.1)

Examples 
----------

.. nbgallery::

   ../examples/etp-based-phase-estimation

References
-----------
.. bibliography:: ../references.bib
    :style: unsrt