import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.patches import Circle
import numpy as np
import pytest

from EEGPhasePy.viz import plot_topomap


def test_plot_topomap_averages_trigger_values():
    topomap_data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
    ])
    positions = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    figure = plot_topomap(topomap_data, positions)
    try:
        assert isinstance(figure, matplotlib.figure.Figure)
        np.testing.assert_array_equal(
            figure.axes[0].collections[-1].get_array(),
            [2.0, 3.0, 4.0, 5.0])
        assert any(
            isinstance(patch, Circle) for patch in figure.axes[0].patches)
    finally:
        plt.close(figure)


def test_plot_topomap_rejects_mismatched_positions():
    with pytest.raises(ValueError, match='must have shape'):
        plot_topomap(np.ones((2, 4)), np.ones((3, 2)))
