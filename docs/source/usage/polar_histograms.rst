Polar histograms
================

EEGPhasePy provides a helper function to plot the distribution of trigger
phases as a polar histogram. The function returns a
:py:class:`matplotlib.figure.Figure` object, so you can continue customizing
it after creation.

Basic usage
-----------

If you already have an EEGPhasePy estimator fitted to your data, use
:py:meth:`~EEGPhasePy.estimators.Estimator.polar_histogram_from_triggers`.
Pass your raw EEG array and the sample indices where each trigger fired:

.. code-block:: python
   :caption: Plotting with an existing estimator

   polar_hist_fig = etp.polar_histogram_from_triggers(testing_signal, triggers)

You can also plot directly from an array of phase values (in radians) using
:py:func:`EEGPhasePy.viz.plot_polar_histogram` — useful when you have already
computed phases outside of an estimator:

.. plot::
   :include-source:
   :caption: Basic polar histogram

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 200, 1 / fs)
   signal_clean = np.sin(2 * np.pi * 10 * time_data)

   peaks = signal.find_peaks(signal_clean)[0]
   phase_data = np.angle(signal.hilbert(signal_clean))

   rng = np.random.default_rng(0)
   fig = viz.plot_polar_histogram(phase_data[peaks] + rng.normal(0, 0.7, len(peaks)))

Adjusting bin width
-------------------

The ``bin_width`` parameter controls how wide each bar is in degrees. Smaller
values give a finer-grained histogram; larger values give a coarser summary:

.. plot::
   :include-source:
   :caption: Narrow bins (10°)

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 200, 1 / fs)
   signal_clean = np.sin(2 * np.pi * 10 * time_data)

   peaks = signal.find_peaks(signal_clean)[0]
   phase_data = np.angle(signal.hilbert(signal_clean))

   rng = np.random.default_rng(0)
   phases = phase_data[peaks] + rng.normal(0, 0.7, len(peaks))

   fig = viz.plot_polar_histogram(phases, bin_width=10)

.. plot::
   :include-source:
   :caption: Default bins (22.5°)

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 200, 1 / fs)
   signal_clean = np.sin(2 * np.pi * 10 * time_data)

   peaks = signal.find_peaks(signal_clean)[0]
   phase_data = np.angle(signal.hilbert(signal_clean))

   rng = np.random.default_rng(0)
   phases = phase_data[peaks] + rng.normal(0, 0.7, len(peaks))

   fig = viz.plot_polar_histogram(phases, bin_width=22.5)

Color customization
-------------------

:py:func:`~EEGPhasePy.viz.plot_polar_histogram` accepts ``color`` for the bar
fill and ``edge_color`` for the bar outlines. Both accept any
:py:mod:`matplotlib` color: a named string, a hex code, or an RGB tuple.

.. plot::
   :include-source:
   :caption: Custom fill and outline colors

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 200, 1 / fs)
   signal_clean = np.sin(2 * np.pi * 10 * time_data)

   peaks = signal.find_peaks(signal_clean)[0]
   phase_data = np.angle(signal.hilbert(signal_clean))

   rng = np.random.default_rng(0)
   phases = phase_data[peaks] + rng.normal(0, 0.7, len(peaks))

   # Coral fill with a dark-red outline
   fig = viz.plot_polar_histogram(
       phases,
       color="#E8735A",
       edge_color="#8B1A0E",
   )

Advanced matplotlib customization
----------------------------------

The function returns a standard :py:class:`matplotlib.figure.Figure`, so you
can reach into its axes and apply any matplotlib customization after the fact.

**Post-hoc axes edits**

.. plot::
   :include-source:
   :caption: Adding a title and hiding gridlines

   import numpy as np
   import scipy.signal as signal
   import EEGPhasePy.viz as viz

   fs = 2000
   time_data = np.arange(0, 200, 1 / fs)
   signal_clean = np.sin(2 * np.pi * 10 * time_data)

   peaks = signal.find_peaks(signal_clean)[0]
   phase_data = np.angle(signal.hilbert(signal_clean))

   rng = np.random.default_rng(0)
   phases = phase_data[peaks] + rng.normal(0, 0.7, len(peaks))

   fig = viz.plot_polar_histogram(phases)

   ax = fig.axes[0]
   ax.set_title("Phase distribution at trigger — C3", pad=14)
   ax.yaxis.grid(False)
   ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

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
   time_data = np.arange(0, 200, 1 / fs)
   signal_clean = np.sin(2 * np.pi * 10 * time_data)

   peaks = signal.find_peaks(signal_clean)[0]
   phase_data = np.angle(signal.hilbert(signal_clean))

   rng = np.random.default_rng(0)
   phases = phase_data[peaks] + rng.normal(0, 0.7, len(peaks))

   plt.style.use("ggplot")
   fig = viz.plot_polar_histogram(phases)
   plt.style.use("default")  # reset so other plots in the session are unaffected

.. note::

   Because the figure is created *inside* the function, you cannot pass a
   pre-existing :py:class:`~matplotlib.axes.Axes` to embed the polar histogram
   in a larger subplot grid. To work around this, create your subplot grid
   first, call :py:func:`~EEGPhasePy.viz.plot_polar_histogram` to get the
   figure, copy its axes content, and paste it into your target axes — or
   simply save the returned figure and assemble a composite image afterwards.
