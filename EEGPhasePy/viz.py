'''
EEGPhasePy visualization module

The visualization module consists of helper functions for plotting
phase histograms and pre-trigger/post-trigger waveforms
'''
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .utils.check import _check_type, _check_array_dimensions


def plot_polar_histogram(phase_data: npt.ArrayLike,
                         bin_width=22.5,
                         color=(3/255, 148/255, 252/255),
                         edge_color=(0/255, 98/255, 255/255)) -> plt.Figure:
    '''
    Plot the polar histogram for an array of phases

    Parameters
    -----------
    phase_data : ndarray[float | int]
        An array of floats containing the phases to construct a histogram out
        of. Must be in radians
    bin_width : float | int
        The width of the bin in degrees
    color : color
        The fill color of the histogram bars. Accepts any matplotlib color
        format (named color, hex string, or RGB tuple). Defaults to blue.
    edge_color : color
        The edge/outline color of the histogram bars. Accepts any matplotlib
        color format. Defaults to dark blue.

    Returns
    --------
    figure : matplotlib.pyplot.Figure
        The figure object for the polar histogram
    '''
    _check_type(phase_data, ['array'])
    _check_type(bin_width, ['int', 'float'])

    _check_array_dimensions(phase_data, [(1,)])

    fig = plt.figure()

    deg_phase = np.rad2deg(phase_data)
    degrees_full_circle = [phase % 360 for phase in deg_phase]

    bin_size = bin_width
    a, b = np.histogram(degrees_full_circle,
                        bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b) / 2 + b[:-1])

    ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, alpha=0.8,
           facecolor=color,
           edgecolor=edge_color,
           linewidth=2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    return fig


def plot_waveform_average(waveforms: npt.ArrayLike,
                          fs: int,
                          t_trigger: int,
                          show_std=True,
                          color=(0/255, 98/255, 255/255),
                          std_color=None) -> plt.Figure:
    '''
    Plot the average waveform and standard deviation as a highlight around the
    mean waveform

    Parameters
    ----------
    waveforms : array_like (n_trigger_segments, n_time)
        A 2D array containing some pre-trigger and post-trigger waveform
        content
    fs : int
        The sampling rate of the data
    t_trigger : int
        The sample number that corresponds to `t = 0`
    show_std : bool
        Defaults to True. Whether to show the standard deviation highlight
    color : color
        The color of the mean waveform line and (by default) the std shading.
        Accepts any matplotlib color format (named color, hex string, or RGB
        tuple). Defaults to blue.
    std_color : color | None
        The color of the standard deviation shading. If None, uses ``color``.
        Accepts any matplotlib color format.

    Returns
    --------
    waveform_figure : matplotlib.pyplot.Figure
        The waveform figure object
    '''
    _check_type(waveforms, ['array'])
    _check_type(show_std, ['bool'])

    _check_array_dimensions(waveforms, [(1, 1)])

    waveforms = np.array(waveforms)
    if np.max(waveforms) < 5e-5:
        waveform_data = waveforms * 1e6
    else:
        waveform_data = waveforms

    mean_waveform = np.mean(waveform_data, axis=0)
    waveform_std = np.std(waveform_data, axis=0)

    time_start = (len(mean_waveform) - t_trigger) / fs
    time_end = (len(mean_waveform) - fs*time_start) / fs
    time_data = np.arange(-time_start, time_end, 1/fs)

    _std_color = std_color if std_color is not None else color

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_data, mean_waveform, color=color, alpha=1)
    if show_std:
        ax.fill_between(time_data, mean_waveform - waveform_std, mean_waveform +
                        waveform_std, alpha=0.2, color=_std_color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV)")

    return fig
