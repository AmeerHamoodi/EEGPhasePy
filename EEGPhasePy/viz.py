'''
EEGPhasePy visualization module

The visualization module consists of helper functions for plotting
phase histograms and pre-trigger/post-trigger waveforms
'''
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from scipy.interpolate import RBFInterpolator

from .utils.check import (
    _check_array_dimensions,
    _check_real_array,
    _check_type,
)


def plot_topomap(topomap_data: npt.ArrayLike,
                 positions: npt.ArrayLike) -> plt.Figure:
    '''
    Plot the average voltage topography across trigger events

    The voltage field is smoothly interpolated between channel positions and
    displayed within a schematic head outline.

    Parameters
    ----------
    topomap_data : array_like (n_triggers, n_channels)
        Voltage values at t=0 for each trigger and channel
    positions : array_like (n_channels, 2)
        The 2D location of each channel, in the same order as
        ``topomap_data`` columns

    Returns
    -------
    figure : matplotlib.pyplot.Figure
        The figure object for the average voltage topography
    '''
    topomap_data = _check_real_array(
        topomap_data, 'topomap_data', contents='voltage values')
    positions = _check_real_array(positions, 'positions',
                                  contents='coordinates')
    _check_array_dimensions(topomap_data, [(1, 1)])
    _check_array_dimensions(positions, [(1, 1)])

    if topomap_data.shape[0] == 0:
        raise ValueError('topomap_data must contain at least one trigger')
    if topomap_data.shape[1] < 3:
        raise ValueError('At least three channel positions are required')
    if positions.shape != (topomap_data.shape[1], 2):
        raise ValueError(
            'positions must have shape (n_channels, 2) and match '
            'topomap_data')
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError('positions must not contain duplicate coordinates')
    if np.linalg.matrix_rank(positions - positions[0]) < 2:
        raise ValueError('positions must not be collinear')

    mean_voltage = np.mean(topomap_data, axis=0)
    if not np.all(np.isfinite(mean_voltage)):
        raise ValueError('average voltage values must be finite')

    color_limit = np.max(np.abs(mean_voltage))
    if color_limit == 0:
        color_limit = 1
    levels = np.linspace(-color_limit, color_limit, 64)

    centered_positions = positions - positions.mean(axis=0)
    position_radius = np.max(np.linalg.norm(centered_positions, axis=1))
    head_radius = 1.0
    sensor_positions = centered_positions * (0.9 / position_radius)

    grid_axis = np.linspace(-1.1, 1.1, 128)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    interpolator = RBFInterpolator(
        sensor_positions, mean_voltage, kernel='thin_plate_spline')
    voltage_grid = interpolator(grid_points).reshape(grid_x.shape)
    outside_head = grid_x**2 + grid_y**2 > head_radius**2
    voltage_grid[outside_head] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(
        grid_x, grid_y, voltage_grid,
        levels=levels, cmap='RdYlBu_r', extend='both')
    ax.contour(
        grid_x, grid_y, voltage_grid,
        levels=np.linspace(-color_limit, color_limit, 9),
        colors='black', linewidths=0.45, alpha=0.65)
    ax.scatter(
        sensor_positions[:, 0], sensor_positions[:, 1],
        c=mean_voltage, cmap='RdYlBu_r',
        vmin=-color_limit, vmax=color_limit,
        edgecolors='black', linewidths=0.4, s=14)

    ax.add_patch(Circle(
        (0, 0), head_radius, fill=False, edgecolor='black', linewidth=1.2))
    ax.plot([-0.08, 0, 0.08], [head_radius, 1.09, head_radius],
            color='black', linewidth=1.2)
    ear_vertices = [
        (-head_radius, 0.18),
        (-1.09, 0.14),
        (-1.09, -0.14),
        (-head_radius, -0.18),
    ]
    ear_codes = [
        Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
    ]
    for direction in (-1, 1):
        vertices = [(direction * x, y) for x, y in ear_vertices]
        ax.add_patch(PathPatch(
            Path(vertices, ear_codes),
            fill=False, edgecolor='black', linewidth=1.2))

    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()
    fig.colorbar(contour, ax=ax, label='Voltage')

    return fig


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

    time_data = (np.arange(len(mean_waveform)) - t_trigger) / fs

    _std_color = std_color if std_color is not None else color

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_data, mean_waveform, color=color, alpha=1)
    if show_std:
        ax.fill_between(time_data, mean_waveform - waveform_std, mean_waveform
                        + waveform_std, alpha=0.2, color=_std_color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV)")

    return fig
