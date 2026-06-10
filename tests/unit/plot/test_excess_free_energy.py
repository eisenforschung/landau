"""Tests for plot_excess_free_energy."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd
import pytest
import seaborn as sns

from landau import IdealSolution, LinePhase
from landau.calculate import calc_phase_diagram
from landau.phases import kB
from landau.plot import plot_excess_free_energy


def _make_phases():
    """Two-component system: ideal solid solution + intermetallic line phase."""
    e0 = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    e1 = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    solid = IdealSolution("solid", e0, e1)
    inter = LinePhase("AB", fixed_concentration=0.5, line_energy=-2.8, line_entropy=1.3 * kB)
    return [solid, inter]


def _minimal_df(temperatures, n_points=50):
    """Pre-compute DataFrame for the minimal two-component system."""
    phases = _make_phases()
    try:
        temps = list(temperatures)
    except TypeError:
        temps = [float(temperatures)]
    return pd.concat(
        [calc_phase_diagram(phases, Ts=T, mu=n_points, keep_unstable=True) for T in temps],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Return type and shape
# ---------------------------------------------------------------------------


def test_multi_temperature_returns_facetgrid():
    g = plot_excess_free_energy(_minimal_df([500, 1000, 1500]))
    assert isinstance(g, sns.FacetGrid)
    assert isinstance(g.fig, plt.Figure)
    plt.close(g.fig)


def test_single_temperature_returns_axes():
    ax = plot_excess_free_energy(_minimal_df([1000]))
    assert isinstance(ax, plt.Axes)
    assert not isinstance(ax, sns.FacetGrid)
    plt.close(ax.figure)


def test_axes_has_one_entry_per_temperature():
    g = plot_excess_free_energy(_minimal_df([500, 1000, 1500]))
    assert g.axes.size == 3
    plt.close(g.fig)


def test_empty_df_raises():
    empty = pd.DataFrame(columns=["c", "f_excess", "phase", "T", "stable"])
    with pytest.raises(ValueError, match="empty"):
        plot_excess_free_energy(empty)


def test_scalar_temperature_accepted():
    ax = plot_excess_free_energy(_minimal_df(1000))
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# convex_hull flag
# ---------------------------------------------------------------------------


def test_convex_hull_true_adds_hull_line():
    ax = plot_excess_free_energy(_minimal_df([1000]), convex_hull=True)
    black_dotted = [l for l in ax.lines if l.get_color() == "k" and l.get_linestyle() == ":"]
    assert len(black_dotted) >= 1
    plt.close(ax.figure)


def test_convex_hull_false_has_no_hull_line():
    ax = plot_excess_free_energy(_minimal_df([1000]), convex_hull=False)
    black_dotted = [l for l in ax.lines if l.get_color() == "k" and l.get_linestyle() == ":"]
    assert len(black_dotted) == 0
    plt.close(ax.figure)


def test_convex_hull_true_adds_extra_scatter():
    """convex_hull=True should add hull vertex scatter on top of line-phase scatter."""
    df = _minimal_df([1000])
    ax_no = plot_excess_free_energy(df, convex_hull=False)
    n_no = len(list(ax_no.collections))
    plt.close(ax_no.figure)

    ax_yes = plot_excess_free_energy(df, convex_hull=True)
    n_yes = len(list(ax_yes.collections))
    plt.close(ax_yes.figure)

    assert n_yes > n_no


# ---------------------------------------------------------------------------
# Line phase rendering
# ---------------------------------------------------------------------------


def test_line_phase_renders_as_scatter():
    """Line phase (AB at c=0.5) must appear as a scatter dot, not as a curve."""
    ax = plot_excess_free_energy(_minimal_df([1000]), convex_hull=False)
    assert len(ax.collections) >= 1
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Single temperature — comprehensive
# ---------------------------------------------------------------------------


def test_single_temperature_convex_hull_true():
    """Single-T DataFrame: hull line and vertex scatter must be present."""
    ax = plot_excess_free_energy(_minimal_df([1000]), convex_hull=True)
    black_dotted = [l for l in ax.lines if l.get_color() == "k" and l.get_linestyle() == ":"]
    assert len(black_dotted) >= 1, "hull tangent line missing for single temperature"
    assert len(ax.collections) >= 1, "hull vertex scatter missing for single temperature"
    plt.close(ax.figure)


def test_single_temperature_draws_on_current_axes():
    """One temperature draws on the caller's current axes instead of a new figure."""
    fig, ax = plt.subplots()
    ret = plot_excess_free_energy(_minimal_df([1000]))
    assert ret is ax
    plt.close(fig)


def test_all_line_phases_no_crash():
    """When all phases are line phases, plot_excess_free_energy must not crash."""
    from landau.calculate import calc_phase_diagram

    e0 = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    e1 = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    inter = LinePhase("AB", fixed_concentration=0.5, line_energy=-2.8, line_entropy=1.3 * kB)
    df = calc_phase_diagram([e0, e1, inter], Ts=1000, mu=50, keep_unstable=True)
    ax = plot_excess_free_energy(df, convex_hull=True)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_all_line_phases_legend_present():
    """With inline_legend=False and only line phases the figure legend lists all names."""
    from landau.calculate import calc_phase_diagram

    e0 = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    e1 = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    inter = LinePhase("AB", fixed_concentration=0.5, line_energy=-2.8, line_entropy=1.3 * kB)
    df = calc_phase_diagram([e0, e1, inter], Ts=1000, mu=50, keep_unstable=True)
    ax = plot_excess_free_energy(df, convex_hull=True, inline_legend=False)
    legend = ax.figure.legends
    assert legend, "figure legend is missing when only line phases are stable"
    legend_labels = {t.get_text() for t in legend[0].texts}
    assert {"A", "B", "AB"} <= legend_labels
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Inline labels
# ---------------------------------------------------------------------------


def test_inline_legend_default_drops_figure_legend():
    """Inline labels are on by default and replace the figure legend box."""
    ax = plot_excess_free_energy(_minimal_df([1000]))
    assert not ax.figure.legends
    assert ax.get_legend() is None
    plt.close(ax.figure)


def test_inline_legend_labels_every_phase_on_each_facet():
    """Each facet carries one inline text label per phase (solution + line)."""
    g = plot_excess_free_energy(_minimal_df([500, 1000]))
    for ax in g.axes.flat:
        labels = {t.get_text() for t in ax.texts}
        # 'solid' is a solution curve, 'AB' a line-phase dot; both must be labelled.
        assert {"solid", "AB"} <= labels
    plt.close(g.fig)


def test_inline_label_colors_match_curves():
    """Inline label colours match the phase palette, not a default black."""
    df = _minimal_df([1000])
    override = {"solid": "#123456"}
    ax = plot_excess_free_energy(df, color_override=override)
    solid_label = next(t for t in ax.texts if t.get_text() == "solid")
    assert to_rgba(solid_label.get_color()) == to_rgba("#123456")
    plt.close(ax.figure)


def test_inline_labels_clear_of_curves_and_markers():
    """Every inline label box must clear the drawn curves and scatter markers."""
    import shapely

    from landau.plot import _curve_obstacles

    g = plot_excess_free_energy(_minimal_df([500, 1000]))
    g.fig.canvas.draw()
    renderer = g.fig.canvas.get_renderer()
    for ax in g.axes.flat:
        obstacles = _curve_obstacles(ax)
        assert obstacles is not None, "expected curves/markers to build obstacles"
        for t in ax.texts:
            e = t.get_window_extent(renderer)
            box = shapely.box(e.x0, e.y0, e.x1, e.y1)
            assert not box.intersects(obstacles), f"label {t.get_text()!r} overlaps a curve/marker"
    plt.close(g.fig)


def test_inline_legend_false_keeps_figure_legend():
    """inline_legend=False keeps the right-hand figure legend and adds no inline text."""
    ax = plot_excess_free_energy(_minimal_df([1000]), inline_legend=False)
    assert ax.figure.legends
    assert not ax.texts
    plt.close(ax.figure)
