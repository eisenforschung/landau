"""Unit tests for _plot_tielines (unrefined and refined branches).

_plot_tielines is called from plot_phase_diagram(tielines=True) and
handles two code paths depending on whether a 'refined' column is present:

* Unrefined branch: detects triple-point temperatures by counting distinct
  phases per T and draws ax.plot segments at each T where the count changes
  and the (T, mu) group contains exactly two phases.
* Refined branch: calls get_transitions() to obtain border_segment groups and
  draws ax.hlines for segments with 3+ distinct phases (triple points).

These tests pin the current behaviour of each branch so a future refactor
(splitting the two plot_tie definitions into separate helpers) cannot change
render output silently.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import pytest

import landau.calculate as ldc
import landau.phases as ldp
from landau.plot import _plot_tielines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hline_collections(ax):
    """LineCollection objects added by ax.hlines (the refined-branch path)."""
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def _tieline_lines(ax):
    """Line2D objects with exactly 2 data points (the unrefined-branch path)."""
    return [l for l in ax.lines if len(l.get_xdata()) == 2]


# ---------------------------------------------------------------------------
# Unrefined branch
# ---------------------------------------------------------------------------


def test_tielines_unrefined_draws_segment_at_count_change():
    """Unrefined branch: one ax.plot tie-line at the T where per-T phase count changes.

    T=500: one row (phase A) → count 1.
    T=600: two rows (A at c=0.2, B at c=0.8) → count 2, diff=1 → T_tie=[600].
    The (T=600, mu=0) group has exactly 2 rows, so ax.plot fires once.
    The segment must run from c=0.2 to c=0.8 at T=600.
    """
    df = pd.DataFrame({
        "T":     [500.0, 600.0, 600.0],
        "mu":    [  0.0,   0.0,   0.0],
        "c":     [  0.2,   0.2,   0.8],
        "phase": [  "A",   "A",   "B"],
    })
    fig, ax = plt.subplots()
    try:
        _plot_tielines(df, ax=ax)
        lines = _tieline_lines(ax)
        assert len(lines) == 1, f"expected 1 tie-line, got {len(lines)}"
        np.testing.assert_allclose(sorted(lines[0].get_xdata()), [0.2, 0.8])
        np.testing.assert_allclose(lines[0].get_ydata(), [600.0, 600.0])
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Refined branch — fixtures
# ---------------------------------------------------------------------------

# hcp / fcc / liquid ideal-solution system from Basics.ipynb.
# At Ts ∈ [200, 1000] K and auto-detected mu range this system has a
# triple point near T ≈ 780 K where hcp, fcc, and liquid all coexist.
# The triple-point concentrations are close enough (~0.24 / 0.34 / 0.51)
# that cluster_T_c_mu merges them into one border_segment with 3 phases,
# which causes plot_tie to call ax.hlines.

_FCCA = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB)
_FCCB = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * ldp.kB)
_HCPA = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
_HCPB = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
_LQDA = ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * ldp.kB)
_LQDB = ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * ldp.kB)
_FCC = ldp.IdealSolution("fcc", _FCCA, _FCCB)
_HCP = ldp.IdealSolution("hcp", _HCPA, _HCPB)
_LQD = ldp.IdealSolution("liquid", _LQDA, _LQDB)


@pytest.fixture(scope="module")
def df_two_phase_refined():
    """Refined c-T diagram for hcp + liquid only (no fcc).

    Two-phase system: no triple point exists, so every border_segment group
    contains at most 2 distinct phases and plot_tie always returns early
    without calling ax.hlines.
    """
    Ts = np.linspace(200.0, 1000.0, 20)
    return ldc.calc_phase_diagram([_HCP, _LQD], Ts, mu=30, refine=True)


@pytest.fixture(scope="module")
def df_triple_point_refined():
    """Refined c-T diagram for hcp + fcc + liquid (Basics.ipynb system).

    Three-phase system with a eutectic triple point near T ≈ 780 K.
    The DelaunayTripleRefiner produces three border rows at the triple point;
    get_transitions clusters them into one border_segment whose plot_tie call
    reaches ax.hlines (3 distinct phases in the group, not in [1, 2]).
    """
    Ts = np.linspace(200.0, 1000.0, 25)
    return ldc.calc_phase_diagram([_HCP, _FCC, _LQD], Ts, mu=50, refine=True)


# ---------------------------------------------------------------------------
# Refined branch — tests
# ---------------------------------------------------------------------------


def test_tielines_refined_two_phase_no_hlines(df_two_phase_refined):
    """Refined branch: two-phase diagram → all border_segment groups have ≤2 phases
    → plot_tie returns early → no ax.hlines calls.
    """
    df = df_two_phase_refined
    assert "refined" in df.columns, "fixture must have 'refined' column so the refined branch is taken"
    assert df["border"].any(), "fixture must have border rows for get_transitions to process"
    fig, ax = plt.subplots()
    try:
        _plot_tielines(df, ax=ax)
        assert _hline_collections(ax) == [], (
            "no triple-point hlines expected for a two-phase system"
        )
    finally:
        plt.close(fig)


def test_tielines_refined_triple_point_draws_hlines(df_triple_point_refined):
    """Refined branch: eutectic triple point → border_segment with 3 phases
    → ax.hlines fired at least once at the triple-point temperature.

    The hline sits within 30 K of the known triple-point temperature (~780 K).
    """
    df = df_triple_point_refined
    assert "refined" in df.columns, "fixture must have 'refined' column"
    triple_rows = df[df["refined"] == "delaunay-triple"]
    assert not triple_rows.empty, "fixture must contain delaunay-triple rows"
    T_triple = triple_rows["T"].mean()

    fig, ax = plt.subplots()
    try:
        _plot_tielines(df, ax=ax)
        lcs = _hline_collections(ax)
        assert len(lcs) >= 1, "expected at least one ax.hlines at the triple point"
        # Verify at least one hline sits near the known triple-point temperature.
        triple_lcs = [
            lc for lc in lcs
            if any(abs(seg[0, 1] - T_triple) < 30.0 for seg in lc.get_segments())
        ]
        assert triple_lcs, (
            f"no hline found within 30 K of T_triple={T_triple:.1f} K; "
            f"hline y-coords: {[lc.get_segments()[0][0, 1] for lc in lcs]}"
        )
    finally:
        plt.close(fig)
