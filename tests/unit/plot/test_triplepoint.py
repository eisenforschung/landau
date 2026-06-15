"""Integration tests for _plot_triplepoint against real refined diagrams.

_plot_triplepoint is called from plot_phase_diagram(plot_triplepoint=True) and
draws one isothermal line per three-phase invariant, reading the `locus` column
(landau.features.Locus.TRIPLE) of a refined calc_phase_diagram frame. These
tests check the end-to-end chain on real systems: a two-phase diagram has no
triple point and draws nothing, a three-phase diagram draws a line at the
eutectic temperature. The synthetic per-row behaviour is pinned in
test_polygons.py.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pytest

import landau.calculate as ldc
import landau.phases as ldp
from landau.features import Locus
from landau.plot import _plot_triplepoint


def _hline_collections(ax):
    """LineCollection objects added by ax.hlines."""
    return [c for c in ax.collections if isinstance(c, LineCollection)]


# hcp / fcc / liquid ideal-solution system from Basics.ipynb, with a eutectic
# triple point where hcp, fcc, and liquid coexist.
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
    """Refined c-T diagram for hcp + liquid only — no triple point exists."""
    Ts = np.linspace(200.0, 1000.0, 20)
    return ldc.calc_phase_diagram([_HCP, _LQD], Ts, mu=30, refine=True)


@pytest.fixture(scope="module")
def df_triple_point_refined():
    """Refined c-T diagram for hcp + fcc + liquid with a eutectic triple point."""
    Ts = np.linspace(200.0, 1000.0, 25)
    return ldc.calc_phase_diagram([_HCP, _FCC, _LQD], Ts, mu=50, refine=True)


def test_triplepoint_two_phase_draws_nothing(df_two_phase_refined):
    """A two-phase diagram has no Locus.TRIPLE rows, so no line is drawn."""
    df = df_two_phase_refined
    assert (df["locus"] != Locus.TRIPLE).all(), "fixture must have no triple point"
    fig, ax = plt.subplots()
    try:
        _plot_triplepoint(df, ax=ax)
        assert _hline_collections(ax) == []
    finally:
        plt.close(fig)


def test_triplepoint_three_phase_draws_line_at_eutectic(df_triple_point_refined):
    """A eutectic triple point draws an isothermal line at its temperature.

    The line spans the concentrations of the three coexisting phases and sits at
    the temperature shared by the Locus.TRIPLE rows.
    """
    df = df_triple_point_refined
    triple = df[df["locus"] == Locus.TRIPLE]
    assert not triple.empty, "fixture must contain a triple point"
    # All triple rows share one invariant (T, mu).
    T_triple = triple["T"].mean()
    cmin, cmax = triple["c"].min(), triple["c"].max()

    fig, ax = plt.subplots()
    try:
        _plot_triplepoint(df, ax=ax)
        lcs = _hline_collections(ax)
        assert len(lcs) == 1, "expected exactly one invariant line"
        seg = lcs[0].get_segments()[0]
        (x0, y0), (x1, y1) = seg
        assert y0 == y1 == pytest.approx(T_triple)
        assert (min(x0, x1), max(x0, x1)) == pytest.approx((cmin, cmax))
    finally:
        plt.close(fig)
