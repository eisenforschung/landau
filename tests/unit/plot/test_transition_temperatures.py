"""Tests for transition-temperature annotations on 2d phase diagrams.

_annotate_transition_temperatures (plot_phase_diagram / plot_mu_phase_diagram's
transition_temperatures=True) labels every triple point with its temperature --
these are tagged in the dataframe (Locus.TRIPLE) -- and then does the same for
whatever congruent-transformation points _find_congruent_points turns up.
Congruent points are not tagged, so they are found heuristically: a strict
interior local minimum, below `tol`, of the concentration gap between the two
coexisting phases along a refined boundary_id line.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from landau.features import Locus
from landau.plot import (
    _annotate_transition_temperatures,
    _find_congruent_points,
    plot_phase_diagram,
)
from landau.poly import Concave


@pytest.fixture
def ax():
    """A fresh axes; its figure is closed on teardown."""
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _boundary_df(gaps, Ts, mus=None, phases=("S", "L"), boundary_id=0):
    """A two-phase refined boundary_id line with a given concentration gap per T.

    `gaps[i]` is the concentration gap between the two coexisting phases at
    `Ts[i]` (`mus[i]` if given, else 0, 1, 2, ...), centered on c=0.5.
    """
    if mus is None:
        mus = [float(i) for i in range(len(Ts))]
    lo, hi = phases
    rows = []
    for mu, T, gap in zip(mus, Ts, gaps):
        c_lo, c_hi = 0.5 - gap / 2, 0.5 + gap / 2
        rows.append({"mu": mu, "T": T, "c": c_lo, "phase": lo, "locus": Locus.BOUNDARY, "boundary_id": boundary_id})
        rows.append({"mu": mu, "T": T, "c": c_hi, "phase": hi, "locus": Locus.BOUNDARY, "boundary_id": boundary_id})
    return pd.DataFrame(rows)


@pytest.fixture
def triple_df():
    """Two triple points, as in test_triplepoint.py's fixture of the same name."""
    return pd.DataFrame(
        {
            "mu": [0.2, 0.2, 0.2, -0.1, -0.1, -0.1],
            "T": [300.0, 300.0, 300.0, 450.0, 450.0, 450.0],
            "c": [0.1, 0.5, 0.9, 0.2, 0.4, 0.7],
            "phase": ["A", "B", "C", "A", "B", "C"],
            "locus": [Locus.TRIPLE] * 6,
        }
    )


# --- _find_congruent_points --------------------------------------------------


def test_finds_interior_local_minimum():
    df = _boundary_df(gaps=[0.30, 0.02, 0.30], Ts=[300.0, 320.0, 340.0])
    out = _find_congruent_points(df, tol=0.05)
    assert len(out) == 1
    mu, T, c = out[0]
    assert (mu, T, c) == pytest.approx((1.0, 320.0, 0.5))


def test_ignores_minimum_at_the_trace_edge():
    df = _boundary_df(gaps=[0.30, 0.20, 0.10, 0.02], Ts=[300.0, 310.0, 320.0, 330.0])
    assert _find_congruent_points(df, tol=0.05) == []


def test_above_tolerance_not_flagged():
    df = _boundary_df(gaps=[0.30, 0.20, 0.30], Ts=[300.0, 320.0, 340.0])
    assert _find_congruent_points(df, tol=0.05) == []


def test_miscibility_gap_same_phase_excluded():
    """A miscibility gap's two branches share one phase name; its closing gap is
    a consolute point, not a congruent transformation, and must not be flagged."""
    df = _boundary_df(gaps=[0.30, 0.02, 0.30], Ts=[300.0, 320.0, 340.0], phases=("A", "A"))
    assert _find_congruent_points(df, tol=0.05) == []


@pytest.mark.parametrize(
    "transform",
    [lambda df: df.drop(columns="boundary_id"), lambda df: df.drop(columns="locus")],
    ids=["no-boundary_id", "no-locus"],
)
def test_missing_columns_returns_empty(transform):
    df = _boundary_df(gaps=[0.30, 0.02, 0.30], Ts=[300.0, 320.0, 340.0])
    assert _find_congruent_points(transform(df)) == []


def test_boundary_ids_checked_independently():
    a = _boundary_df(gaps=[0.30, 0.02, 0.30], Ts=[300.0, 320.0, 340.0], boundary_id=0)
    b = _boundary_df(gaps=[0.30, 0.20, 0.30], Ts=[300.0, 320.0, 340.0], boundary_id=1)
    out = _find_congruent_points(pd.concat([a, b], ignore_index=True), tol=0.05)
    assert len(out) == 1
    assert out[0][1] == pytest.approx(320.0)


# --- _annotate_transition_temperatures ---------------------------------------


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_labels_every_triple_point(ax, triple_df, variables):
    _annotate_transition_temperatures(triple_df, ax=ax, variables=variables)
    assert sorted(t.get_text() for t in ax.texts) == ["300 K", "450 K"]


def test_cT_label_sits_past_the_line_end(ax, triple_df):
    _annotate_transition_temperatures(triple_df, ax=ax, variables=["c", "T"])
    by_text = {t.get_text(): t for t in ax.texts}
    x, y = by_text["300 K"].get_position()
    assert x > 0.9  # nudged past this invariant's c_max=0.9
    assert y == pytest.approx(300.0)
    assert by_text["300 K"].get_ha() == "left"


def test_noop_without_locus_column(ax, triple_df):
    _annotate_transition_temperatures(triple_df.drop(columns="locus"), ax=ax)
    assert list(ax.texts) == []


def test_congruent_point_gets_marker_and_label(ax):
    df = _boundary_df(gaps=[0.30, 0.02, 0.30], Ts=[300.0, 320.0, 340.0])
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    assert [t.get_text() for t in ax.texts] == ["320 K"]
    markers = [line for line in ax.lines if line.get_marker() == "D"]
    assert len(markers) == 1
    assert (markers[0].get_xdata()[0], markers[0].get_ydata()[0]) == pytest.approx((0.5, 320.0))


def test_congruent_point_below_tol_not_labelled(ax):
    df = _boundary_df(gaps=[0.30, 0.20, 0.30], Ts=[300.0, 320.0, 340.0])
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    assert list(ax.texts) == [] and list(ax.lines) == []


# --- end-to-end on a real refined diagram ------------------------------------


def test_plot_phase_diagram_labels_the_triple_point(eutectic_diagram):
    """transition_temperatures=True labels the eutectic's temperature, and marks
    the invariant even though triplepoints itself was not requested."""
    fig, ax = plt.subplots()
    try:
        plot_phase_diagram(
            eutectic_diagram,
            ax=ax,
            poly_method=Concave(drop_interior=False),
            transition_temperatures=True,
            legend=False,
        )
        triple = eutectic_diagram[eutectic_diagram["locus"] == Locus.TRIPLE]
        assert not triple.empty, "fixture must contain a triple point"
        T_t = triple["T"].mean()
        assert f"{T_t:.0f} K" in [t.get_text() for t in ax.texts]
        assert any(
            y0 == y1 for coll in ax.collections for (x0, y0), (x1, y1) in coll.get_segments()
        ), "the triple-point isotherm should be drawn even though triplepoints=False"
    finally:
        plt.close(fig)
