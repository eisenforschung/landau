"""Direct unit tests for `landau.plot._curve_obstacles` (issue #389).

`_curve_obstacles` rasterises the lines and scatter markers of a matplotlib
Axes into one shapely geometry in pixel space; `_add_inline_curve_labels`
tests a candidate label's pixel bounding box against it. These tests pin the
branch structure directly instead of only through the end-to-end pixel-obstacle
sanity check in `test_excess_free_energy.py`.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely

from landau.plot import _curve_obstacles


def test_curve_obstacles_empty_axes_returns_none():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    assert _curve_obstacles(ax) is None
    plt.close(fig)


def test_curve_obstacles_line_becomes_linestring_in_pixel_coords():
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    fig.canvas.draw()

    geom = _curve_obstacles(ax)

    assert geom.geom_type == "LineString"
    expected = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    np.testing.assert_allclose(np.array(geom.coords), expected)
    plt.close(fig)


def test_curve_obstacles_single_point_line_dropped():
    fig, ax = plt.subplots()
    ax.plot([0.0], [0.0])
    fig.canvas.draw()
    assert _curve_obstacles(ax) is None
    plt.close(fig)


def test_curve_obstacles_filters_nonfinite_rows():
    fig, ax = plt.subplots()
    ax.plot([0.0, np.nan, 2.0], [0.0, 1.0, 2.0])
    fig.canvas.draw()

    geom = _curve_obstacles(ax)

    assert geom.geom_type == "LineString"
    expected = ax.transData.transform([(0.0, 0.0), (2.0, 2.0)])
    np.testing.assert_allclose(np.array(geom.coords), expected)
    plt.close(fig)


def test_curve_obstacles_skips_blended_transform_axhline():
    fig, ax = plt.subplots()
    ax.axhline(0.5, color="k")
    fig.canvas.draw()
    assert _curve_obstacles(ax) is None
    plt.close(fig)


def test_curve_obstacles_scatter_becomes_buffered_point():
    fig, ax = plt.subplots(dpi=100)
    ax.scatter([0.5], [0.5], s=100.0)
    fig.canvas.draw()

    geom = _curve_obstacles(ax)

    assert geom.geom_type == "Polygon"
    px, py = ax.transData.transform((0.5, 0.5))
    radius = (np.sqrt(100.0) / 2.0) * fig.dpi / 72.0
    assert geom.contains(shapely.Point(px, py))
    assert geom.contains(shapely.Point(px + radius * 0.9, py))
    assert not geom.contains(shapely.Point(px + radius * 1.5, py))
    plt.close(fig)
