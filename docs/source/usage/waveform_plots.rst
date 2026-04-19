Waveform plots
==============

EEGPhasePy provides a helper function to plot the pre/post-trigger waveform
average with a standard deviation highlight. The function returns a
:py:class:`matplotlib.figure.Figure` object, so you can continue customizing it
after creation.

Basic usage
-----------

If you already have an EEGPhasePy estimator fitted to your data, the simplest
path is :py:meth:`~EEGPhasePy.estimators.Estimator.plot_mean_std_waveform_from_triggers`.
Pass your raw EEG array, the sample indices where each trigger fired, and the
time window (in seconds) to include on either side of t = 0:

.. code-block:: python
   :caption: Plotting with an existing estimator

   waveform_fig = etp.plot_mean_std_waveform_from_triggers(
       testing_signal, triggers, tmin=0.1, tmax=0.1
   )

You can also plot directly from a 2-D array of waveform segments using
:py:func:`EEGPhasePy.viz.plot_waveform_average` — useful when you have
already extracted epochs or want to skip the estimator entirely:

.. plot::
   :include-source:
   :caption: Basic waveform average

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 30, 1 / fs)
   raw = np.sin(2 * np.pi * 10 * time_data) + 0.3 * np.random.default_rng(0).standard_normal(len(time_data))

   peaks = signal.find_peaks(np.sin(2 * np.pi * 10 * time_data))[0]

   win_samples = int(0.1 * fs)
   segments = [
       raw[p - win_samples: p + win_samples]
       for p in peaks
       if p - win_samples >= 0 and p + win_samples < len(raw)
   ]
   waveforms = np.array(segments)

   fig = viz.plot_waveform_average(waveforms, fs, t_trigger=win_samples)

Hiding the standard deviation highlight
----------------------------------------

Pass ``show_std=False`` to suppress the shaded region and show only the mean
waveform:

.. plot::
   :include-source:
   :caption: Mean waveform without standard deviation

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 30, 1 / fs)
   raw = np.sin(2 * np.pi * 10 * time_data) + 0.3 * np.random.default_rng(0).standard_normal(len(time_data))

   peaks = signal.find_peaks(np.sin(2 * np.pi * 10 * time_data))[0]
   win_samples = int(0.1 * fs)
   segments = [
       raw[p - win_samples: p + win_samples]
       for p in peaks
       if p - win_samples >= 0 and p + win_samples < len(raw)
   ]
   waveforms = np.array(segments)

   fig = viz.plot_waveform_average(waveforms, fs, t_trigger=win_samples, show_std=False)

Color customization
-------------------

:py:func:`~EEGPhasePy.viz.plot_waveform_average` accepts ``color`` for the
waveform line and ``std_color`` for the shaded region. Both accept any
:py:mod:`matplotlib` color: a named string, a hex code, or an RGB tuple.
When ``std_color`` is not set it inherits from ``color``.

.. plot::
   :include-source:
   :caption: Custom line and shading colors

   import numpy as np
   import scipy.signal as signal
   import matplotlib.pyplot as plt
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 30, 1 / fs)
   raw = np.sin(2 * np.pi * 10 * time_data) + 0.3 * np.random.default_rng(0).standard_normal(len(time_data))

   peaks = signal.find_peaks(np.sin(2 * np.pi * 10 * time_data))[0]
   win_samples = int(0.1 * fs)
   segments = [
       raw[p - win_samples: p + win_samples]
       for p in peaks
       if p - win_samples >= 0 and p + win_samples < len(raw)
   ]
   waveforms = np.array(segments)

   # Purple waveform with a separate orange std shading
   fig = viz.plot_waveform_average(
       waveforms, fs, t_trigger=win_samples,
       color="#7B2D8B",
       std_color="#F4A300",
   )

Advanced matplotlib customization
----------------------------------

The function returns a standard :py:class:`matplotlib.figure.Figure`, so you
can reach into its axes and apply any matplotlib customization after the fact.

**Post-hoc axes edits**

.. plot::
   :include-source:
   :caption: Customizing the returned figure

   import numpy as np
   import scipy.signal as signal
   import matplotlib.pyplot as plt
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 30, 1 / fs)
   raw = np.sin(2 * np.pi * 10 * time_data) + 0.3 * np.random.default_rng(0).standard_normal(len(time_data))

   peaks = signal.find_peaks(np.sin(2 * np.pi * 10 * time_data))[0]
   win_samples = int(0.1 * fs)
   segments = [
       raw[p - win_samples: p + win_samples]
       for p in peaks
       if p - win_samples >= 0 and p + win_samples < len(raw)
   ]
   waveforms = np.array(segments)

   fig = viz.plot_waveform_average(waveforms, fs, t_trigger=win_samples)

   ax = fig.axes[0]
   ax.set_title("Mu-band waveform average — C3")
   ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="Trigger")
   ax.legend()
   ax.spines[["top", "right"]].set_visible(False)

**Global style sheets**

Apply a :py:func:`matplotlib.pyplot.style.use` call before plotting and the
style will take effect inside the function, since the figure is created on the
call:

.. plot::
   :include-source:
   :caption: Using a matplotlib style sheet

   import numpy as np
   import scipy.signal as signal
   import matplotlib.pyplot as plt
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 30, 1 / fs)
   raw = np.sin(2 * np.pi * 10 * time_data) + 0.3 * np.random.default_rng(0).standard_normal(len(time_data))

   peaks = signal.find_peaks(np.sin(2 * np.pi * 10 * time_data))[0]
   win_samples = int(0.1 * fs)
   segments = [
       raw[p - win_samples: p + win_samples]
       for p in peaks
       if p - win_samples >= 0 and p + win_samples < len(raw)
   ]
   waveforms = np.array(segments)

   plt.style.use("ggplot")
   fig = viz.plot_waveform_average(waveforms, fs, t_trigger=win_samples)
   plt.style.use("default")  # reset so other plots in the session are unaffected

.. note::

   Because the figure is created *inside* the function, you cannot pass a
   pre-existing :py:class:`~matplotlib.axes.Axes` to embed the waveform plot
   in a larger subplot grid. To work around this, create your subplot grid
   first, call :py:func:`~EEGPhasePy.viz.plot_waveform_average` to get the
   figure, copy its axes content, and paste it into your target axes — or
   simply save the returned figure and assemble a composite image afterwards.
