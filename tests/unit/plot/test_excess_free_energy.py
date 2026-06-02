"""Tests for plot_excess_free_energy."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def test_returns_facetgrid():
    g = plot_excess_free_energy(_minimal_df([1000]))
    assert isinstance(g, sns.FacetGrid)
    assert isinstance(g.fig, plt.Figure)
    plt.close(g.fig)


def test_axes_has_one_entry_per_temperature():
    g = plot_excess_free_energy(_minimal_df([500, 1000, 1500]))
    assert g.axes.size == 3
    plt.close(g.fig)


def test_empty_df_raises():
    empty = pd.DataFrame(columns=["c", "f_excess", "phase", "T", "stable"])
    with pytest.raises(ValueError, match="empty"):
        plot_excess_free_energy(empty)


def test_scalar_temperature_accepted():
    g = plot_excess_free_energy(_minimal_df(1000))
    assert g.axes.size == 1
    plt.close(g.fig)


# ---------------------------------------------------------------------------
# convex_hull flag
# ---------------------------------------------------------------------------


def test_convex_hull_true_adds_hull_line():
    g = plot_excess_free_energy(_minimal_df([1000]), convex_hull=True)
    ax = g.axes.flat[0]
    black_dotted = [l for l in ax.lines if l.get_color() == "k" and l.get_linestyle() == ":"]
    assert len(black_dotted) >= 1
    plt.close(g.fig)


def test_convex_hull_false_has_no_hull_line():
    g = plot_excess_free_energy(_minimal_df([1000]), convex_hull=False)
    ax = g.axes.flat[0]
    black_dotted = [l for l in ax.lines if l.get_color() == "k" and l.get_linestyle() == ":"]
    assert len(black_dotted) == 0
    plt.close(g.fig)


def test_convex_hull_true_adds_extra_scatter():
    """convex_hull=True should add hull vertex scatter on top of line-phase scatter."""
    df = _minimal_df([1000])
    g_no = plot_excess_free_energy(df, convex_hull=False)
    n_no = len(list(g_no.axes.flat[0].collections))
    plt.close(g_no.fig)

    g_yes = plot_excess_free_energy(df, convex_hull=True)
    n_yes = len(list(g_yes.axes.flat[0].collections))
    plt.close(g_yes.fig)

    assert n_yes > n_no


# ---------------------------------------------------------------------------
# Line phase rendering
# ---------------------------------------------------------------------------


def test_line_phase_renders_as_scatter():
    """Line phase (AB at c=0.5) must appear as a scatter dot, not as a curve."""
    g = plot_excess_free_energy(_minimal_df([1000]), convex_hull=False)
    ax = g.axes.flat[0]
    assert len(ax.collections) >= 1
    plt.close(g.fig)


# ---------------------------------------------------------------------------
# Single temperature — comprehensive
# ---------------------------------------------------------------------------


def test_single_temperature_convex_hull_true():
    """Single-T DataFrame: hull line and vertex scatter must be present."""
    g = plot_excess_free_energy(_minimal_df([1000]), convex_hull=True)
    ax = g.axes.flat[0]
    black_dotted = [l for l in ax.lines if l.get_color() == "k" and l.get_linestyle() == ":"]
    assert len(black_dotted) >= 1, "hull tangent line missing for single temperature"
    assert len(ax.collections) >= 1, "hull vertex scatter missing for single temperature"
    plt.close(g.fig)


def test_single_temperature_figure_not_wide():
    """Figure width for one temperature must not span more than one column."""
    # default height=3.0, aspect=1.3 → one column ≈ 3.9 in; col_wrap=3 would give 11.7 in.
    g = plot_excess_free_energy(_minimal_df([1000]))
    assert g.fig.get_figwidth() < 2 * 3.0 * 1.3
    plt.close(g.fig)


def test_all_line_phases_no_crash():
    """When all phases are line phases, plot_excess_free_energy must not crash."""
    from landau.calculate import calc_phase_diagram

    e0 = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    e1 = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    inter = LinePhase("AB", fixed_concentration=0.5, line_energy=-2.8, line_entropy=1.3 * kB)
    df = calc_phase_diagram([e0, e1, inter], Ts=1000, mu=50, keep_unstable=True)
    g = plot_excess_free_energy(df, convex_hull=True)
    assert g.axes.size == 1
    plt.close(g.fig)
