"""Tests for plot_excess_free_energy and its helper functions."""
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from landau import LinePhase
from landau.plot import (
    _endmember_free_energy,
    _lower_convex_hull,
    plot_excess_free_energy,
)

ATOL = 1e-12


# ---------------------------------------------------------------------------
# Stub non-line phase
# ---------------------------------------------------------------------------


@dataclass
class _LinearPhase:
    """Minimal solution phase: free_energy = a + b*c (independent of T)."""

    name: str
    a: float = 0.0
    b: float = 0.0
    concentration_range: tuple = (0.0, 1.0)

    def free_energy(self, T, c):
        return self.a + self.b * np.asarray(c, dtype=float)


def _two_endmembers(e0=0.0, e1=0.0):
    return (
        LinePhase(name="e0", fixed_concentration=0.0, line_energy=e0),
        LinePhase(name="e1", fixed_concentration=1.0, line_energy=e1),
    )


# ---------------------------------------------------------------------------
# Group 1: _lower_convex_hull
# ---------------------------------------------------------------------------


def test_hull_flat_line_returns_endpoints():
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    f = [0.0, 0.0, 0.0, 0.0, 0.0]
    c_h, f_h = _lower_convex_hull(c, f)
    assert len(c_h) == 2
    assert abs(c_h[0] - 0.0) < ATOL
    assert abs(c_h[-1] - 1.0) < ATOL


def test_hull_v_shape_includes_minimum():
    c = [0.0, 0.5, 1.0]
    f = [0.0, -1.0, 0.0]
    c_h, f_h = _lower_convex_hull(c, f)
    assert len(c_h) == 3
    np.testing.assert_allclose(c_h, [0.0, 0.5, 1.0], atol=ATOL)
    np.testing.assert_allclose(f_h, [0.0, -1.0, 0.0], atol=ATOL)


def test_hull_interior_above_chord_is_dropped():
    c = [0.0, 0.5, 1.0]
    f = [0.0, 0.5, 0.0]
    c_h, f_h = _lower_convex_hull(c, f)
    assert len(c_h) == 2
    np.testing.assert_allclose(c_h, [0.0, 1.0], atol=ATOL)


def test_hull_ordering_invariance():
    c = [0.0, 0.5, 0.75, 1.0]
    f = [0.0, -0.3, -0.1, 0.0]
    c_fwd, f_fwd = _lower_convex_hull(c, f)
    c_rev, f_rev = _lower_convex_hull(c[::-1], f[::-1])
    np.testing.assert_allclose(c_fwd, c_rev, atol=ATOL)
    np.testing.assert_allclose(f_fwd, f_rev, atol=ATOL)


def test_hull_single_point():
    c_h, f_h = _lower_convex_hull([0.5], [1.0])
    assert len(c_h) == 1
    assert abs(c_h[0] - 0.5) < ATOL


def test_hull_two_points_unchanged():
    c_h, f_h = _lower_convex_hull([0.0, 1.0], [0.0, 0.0])
    assert len(c_h) == 2


# ---------------------------------------------------------------------------
# Group 2: _endmember_free_energy
# ---------------------------------------------------------------------------


def test_endmember_picks_lower_of_two_phases_at_c0():
    p_low = LinePhase(name="low", fixed_concentration=0.0, line_energy=-2.0)
    p_high = LinePhase(name="high", fixed_concentration=0.0, line_energy=-1.0)
    sol = _LinearPhase(name="sol", a=0.5)
    result = _endmember_free_energy([p_low, p_high, sol], T=300, c_end=0.0)
    assert abs(result - (-2.0)) < ATOL


def test_endmember_missing_raises():
    p0 = LinePhase(name="e0", fixed_concentration=0.0, line_energy=0.0)
    with pytest.raises(ValueError, match="c=1"):
        _endmember_free_energy([p0], T=300, c_end=1.0)


def test_endmember_solution_phase_covers_endpoint():
    sol = _LinearPhase(name="sol", a=-1.5, b=0.0)
    result = _endmember_free_energy([sol], T=300, c_end=0.0)
    assert abs(result - (-1.5)) < ATOL


# ---------------------------------------------------------------------------
# Group 3: plot_excess_free_energy
# ---------------------------------------------------------------------------


def _minimal_phases():
    e0, e1 = _two_endmembers(e0=-1.0, e1=-0.8)
    sol = _LinearPhase(name="Sol", a=-1.0, b=0.2)
    return [e0, e1, sol]


def test_returns_figure_and_axes_array():
    phases = _minimal_phases()
    fig, axes = plot_excess_free_energy(phases, temperatures=[300])
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 2
    plt.close(fig)


def test_axes_shape_matches_temperatures_and_max_cols():
    phases = _minimal_phases()
    fig, axes = plot_excess_free_energy(phases, temperatures=[300, 600, 900, 1200], max_cols=3)
    assert axes.shape == (2, 3)
    plt.close(fig)


def test_empty_temperatures_raises():
    phases = _minimal_phases()
    with pytest.raises(ValueError, match="temperatures is empty"):
        plot_excess_free_energy(phases, temperatures=[])


def test_convex_hull_false_has_no_hull_line():
    phases = _minimal_phases()
    fig, axes = plot_excess_free_energy(phases, temperatures=[300], convex_hull=False)
    ax = axes[0, 0]
    hull_lines = [l for l in ax.lines if l.get_label() == "convex hull"]
    assert len(hull_lines) == 0
    plt.close(fig)


def test_convex_hull_true_draws_hull_line():
    phases = _minimal_phases()
    fig, axes = plot_excess_free_energy(phases, temperatures=[300], convex_hull=True)
    ax = axes[0, 0]
    hull_lines = [l for l in ax.lines if l.get_label() == "convex hull"]
    assert len(hull_lines) == 1
    plt.close(fig)


def test_convex_hull_true_marks_hull_vertices_as_scatter():
    phases = _minimal_phases()
    fig, axes = plot_excess_free_energy(phases, temperatures=[300], convex_hull=False)
    ax_no_hull = axes[0, 0]
    n_collections_no_hull = len(ax_no_hull.collections)
    plt.close(fig)

    fig2, axes2 = plot_excess_free_energy(phases, temperatures=[300], convex_hull=True)
    ax_hull = axes2[0, 0]
    n_collections_hull = len(ax_hull.collections)
    plt.close(fig2)

    assert n_collections_hull > n_collections_no_hull


def test_line_phase_renders_as_scatter_not_line():
    e0, e1 = _two_endmembers(e0=-1.0, e1=-0.8)
    sol = _LinearPhase(name="Sol", a=-1.0, b=0.2)
    phases = [e0, e1, sol]

    fig, axes = plot_excess_free_energy(phases, temperatures=[300], convex_hull=False)
    ax = axes[0, 0]
    # Sol is the only solution phase → one line; e0 and e1 are line phases → scatters
    assert len(ax.lines) == 1
    assert len(ax.collections) >= 2  # at least one scatter per line phase
    plt.close(fig)


def test_single_temperature_as_scalar():
    phases = _minimal_phases()
    fig, axes = plot_excess_free_energy(phases, temperatures=300)
    assert axes.shape[1] == 1
    plt.close(fig)
