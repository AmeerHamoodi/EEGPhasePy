Plotting
========================

EEGPhasePy comes with 2 plotting helper methods. We provide helper functions to plot the polar histogram given a set of trigger samples and 
to plot the pre/post-trigger waveform average with standard deviation highlights. Each plotting method returns its :py:mod:`matplotlib` figure object.

All estimators contain stats and plotting helper methods. We will describe how the plotting methods in EEGPhasePy can be used 
with and without an estimator.

Polar histogram 
-----------------
Assuming you have an array, ``triggers``, that contains the sample values at which your target phase was triggered at
you can plot a polar histogram using an existing estimator with the code below:

.. doctest::
   :hide:
   :pyversion: == 3.12


.. code-block:: Python
   :caption: Plotting polar histogram

    polar_hist_fig = etp.polar_histogram_from_triggers(testing_signal, triggers)

If you want to plot a polar histogram without creating an estimator, you can do so by using the :py:mod:`EEGPhasePy.viz` module. See the example code below:

.. testcode::

    import numpy as np
    import scipy.signal as signal

    import EEGPhasePy.viz as viz

    fs = 2000
    time_data = np.arange(0, 200, 1/fs)

    signal_clean = np.sin(2 * np.pi * 10 * time_data)

    peaks = signal.find_peaks(signal_clean)[0]
    phase_data = np.angle(signal.hilbert(signal_clean))

    polar_hist = viz.plot_polar_histogram(phase_data[peaks] + np.random.normal(0, 0.7, len(peaks)))


Waveform average 
------------------
Plotting the average waveform with standard deivation highlights follows similar logic to the polar histogram. With the waveform average 
however, you must specify how much time before and after the stimulus your plot should show. The time before stimulus delievery can be specified
using ``tmin`` (always positive, larger values will encompass a greater amount of pre-stimulus time) and ``tmax``. You can 
plot the pre/post-stimulus waveform given an array of trigger samples and a pre-exisitng estimator using the code below:

.. code-block:: Python
   :caption: Plotting waveform average

    waveform_fig = etp.plot_mean_std_waveform_from_triggers(testing_signal, triggers, tmin=0.1, tmax=0.1)

Unlike the polar histogram, you can only plot waveforms using an existing estimator.

Examples 
----------
.. minigallery:: ../examples/*-phase-estimation.py
