Autoregressive phase estimation
=================================

Background
------------
Autoregressive (AR) phase estimation is a form of phase estimation that uses autoregressive forecasting to fill in the data impacted by 
filter edge effects and tens of milliseconds ahead of the current point in time, to account for Hilbert transform edge effects. AR was first introduced
by :cite:t:`Zrenner2018-hh` and a toolbox for AR phase estimation known as PHASTIMATE was published by :cite:t:`Zrenner2020-zb`

AR is best used in environments that make gaurentees about communication delays as foreward forecasting beyond the current point in time significantly reduces 
AR phase estimation performance. Further, to the best of our knowledge, AR has not been applied as a phase estimation algorithm in environments
where delay gaurentees were made. It is also important to note that AR requires a high packet send rate from your EEG amplifier. This is due to the 
algorithm estimating the current phase. Considering phase changes rapidly across the vast majority of EEG frequencies (except for low-delta)
if the algorithm is only run a couple of times per second, it will very rarely encounter a sliding window whose end contains the desired phase.
The first application of AR ran the algorithm at a frequency of 500Hz (i.e. 500 sliding windows passed into the algorithm / second) :cite:p:`Zrenner2018-hh`.


Usage
---------
AR phase estimation is implemented in the PHASTIMATE class. If you use this class 
please cite :cite:t:`Zrenner2020-zb`. We'll start off by importing the PHASTIMATE class, numpy to simulate EEG data
and scipy.signal for filtering.

.. code-block:: Python
   :caption: Imports
   import numpy as np
   import scipy.signal as signal

   from EEGPhasePy.estimators import PHASTIMATE

Next, we'll create one 200s long signal simulating the human alpha rhythym (assuming 10hz here) with some added gaussian noise.

.. code-block:: Python
   :caption: Simulating data

   fs = 2000
   time_data = np.arange(0, 200, 1/fs)

   signal_clean = np.sin(2 * np.pi * 10 * time_data)
   signal_noisy = training_signal_clean + np.random.normal(0, 2, len(time_data))

After our signals have been created, we need to construct our real-time and ground-truth filter. For simplicity, 
we have assumed that a higher order filter will allow us to effectively obtain ground truth data for most of the signal. 
The purpose of the ground-truth filter is to obtain the true phase. This filter would be used when computing
phase stats (e.g. circular mean and std) based on triggers. The real-time filter is used to filter the raw window
of EEG data passed into the `predict` method.

.. note:: A note on the "ground truth" filter 

   We use ground-truth here quite loosely. It is very challenging to
   obtain a true "ground-truth" for EEG phase. If you're interested in understanding
   more of the nuance associated with this, check out :cite:t:`Zrenner2020-zb`.

.. code-block:: Python
   :caption: Constructing our filters
   
   rt_filter = signal.firwin(120, [8, 12], fs=fs, pass_zero=False)
   gt_filter = signal.firwin(300, [8, 12], fs=fs, pass_zero=False)

Next, we will instantiate PHASTIMATE. 

Unlike ETP, PHASTIMATE doesn't necessarily require training data. With the right parameters
it can work out of the box. However, you may benefit from optimizing the `window_edge` and `ar_order` parameters in the class, especially 
when there is a variability in SNR across participants. Our implementation of the PHASTIMATE toolbox includes genetic optimization in a similar 
manner as what :cite:t:`Zrenner2020-zb` proposed along with a Bayesian Optimization of these parameters. To run an optimization of these parameters 
additional resting-state EEG data would need to be collected, similar to ETP. We have included examples of both optimization approaches at the 
bottom of this page.

.. code-block:: Python
   :caption: Constructing PHASTIMATE
   
   phastimate = PHASTIMATE(rt_filter, gt_filter, fs)

Now, we will run a psuedo-real-time simulation using the testing signal. Here, we use the `predict` method of the `phastimate`
object. `predict` works by returning a boolean value about whether the current phase is your target phase. A tolerance value may also be 
passed in. By default tolerance is 5 degrees. Tolerance indicates how close the current phase must be to the target phase for `predict`
to return `True`.

.. code-block:: Python
   :caption: Psuedo-real-time simulation with PHASTIAMTE

    window_i = 2*fs
    window_len = int(0.5*fs)
    window_step = int(0.06*fs) # using a step of 60 ms

    triggers = []

    while window_i + window_len < len(signal_noisy):
        window = signal_noisy[window_i:window_i + window_len]

        if phastimate.predict(window, 0, 15):
            triggers.append(window_i + window_len)
            
        window_i += window_step

    polar_hist_fig = phastimate.polar_histogram_from_triggers(signal_noisy, triggers)
    waveform_fig = phastimate.plot_mean_std_waveform_from_triggers(signal_noisy, triggers, tmin=0.1, tmax=0.1)

Examples 
----------

.. nbgallery::

   ../examples/ar-phase-estimation
   ../examples/ar-genetic-optimization
   ../examples/ar-bayesian-optimization

References
-----------
.. bibliography:: ../references.bib
    :style: unsrt