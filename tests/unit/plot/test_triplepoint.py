"""Tests for _plot_triplepoints, the triple-point marker for phase diagrams.

_plot_triplepoints is called from plot_phase_diagram(triplepoints=True) (c-T,
isothermal line) and plot_mu_phase_diagram(triplepoints=True) (mu-T, point
marker). It reads the `locus` column (landau.features.Locus.TRIPLE) of a refined
calc_phase_diagram frame: rows sharing one (mu, T) form a three-phase invariant,
drawn as a horizontal line across the coexisting concentrations in c-T and as a
black marker at the (mu, T) in mu-T.

The synthetic tests pin the exact geometry; one end-to-end test confirms a real
refined diagram drives both branches.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import landau.calculate as ldc
import landau.phases as ldp
from landau import plot as plot_mod
from landau.features import Locus
from landau.plot import _plot_triplepoints, plot_phase_diagram
from landau.poly import Concave


def _hlines(ax):
    """(y, xmin, xmax) of every horizontal segment drawn on `ax`."""
    out = []
    for coll in ax.collections:
        for seg in coll.get_segments():
            (x0, y0), (x1, y1) = seg
            if y0 == y1:
                out.append((y0, min(x0, x1), max(x0, x1)))
    return out


def _markers(ax):
    """(x, y) of every single-point marker drawn on `ax`."""
    return [
        (line.get_xdata()[0], line.get_ydata()[0])
        for line in ax.lines
        if len(line.get_xdata()) == 1 and line.get_marker() not in ("", "None", None)
    ]


@pytest.fixture
def ax():
    """A fresh axes; its figure is closed on teardown."""
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture
def triple_df():
    """Frame with two triple points plus boundary/interior distractor rows.

    Invariants at (mu=0.2, T=300) spanning c = 0.1..0.9 and (mu=-0.1, T=450)
    spanning c = 0.2..0.7. The non-TRIPLE rows must be ignored.
    """
    return pd.DataFrame(
        {
            "mu": [0.2, 0.2, 0.2, -0.1, -0.1, -0.1, 0.0, 0.0],
            "T": [300.0, 300.0, 300.0, 450.0, 450.0, 450.0, 250.0, 500.0],
            "c": [0.1, 0.5, 0.9, 0.2, 0.4, 0.7, 0.3, 0.0],
            "phase": ["A", "B", "C", "A", "B", "C", "A", "A"],
            "locus": [Locus.TRIPLE] * 6 + [Locus.BOUNDARY, Locus.INTERIOR],
        }
    )


# --- synthetic geometry ------------------------------------------------------


@pytest.mark.parametrize(
    "variables, lines, markers",
    [
        # c-T (the default axes): one horizontal line per invariant, spanning
        # the coexisting concentrations; no markers.
        (["c", "T"], [(300.0, 0.1, 0.9), (450.0, 0.2, 0.7)], []),
        # mu-T: one marker per invariant (mu, T); no lines.
        (["mu", "T"], [], [(-0.1, 450.0), (0.2, 300.0)]),
    ],
    ids=["c-T", "mu-T"],
)
def test_draws_one_mark_per_invariant(ax, triple_df, variables, lines, markers):
    """Each three-phase invariant is marked once, ignoring non-TRIPLE rows."""
    _plot_triplepoints(triple_df, ax=ax, variables=variables)
    assert sorted(_hlines(ax)) == pytest.approx(lines)
    assert sorted(_markers(ax)) == pytest.approx(markers)
    if markers:  # markers are black
        assert all(line.get_color() in ("k", "black", (0.0, 0.0, 0.0, 1.0)) for line in ax.lines)


@pytest.mark.parametrize(
    "transform",
    [lambda df: df.drop(columns="locus"), lambda df: df.query("locus != 'triple'")],
    ids=["no-locus-column", "no-triple-rows"],
)
def test_noop_without_triple_points(ax, triple_df, transform):
    _plot_triplepoints(transform(triple_df), ax=ax)
    assert _hlines(ax) == [] and _markers(ax) == []


def test_tielines_deprecated_routes_to_triplepoints(ax, monkeypatch):
    """The old `tielines=` keyword still works but warns and maps onto
    `triplepoints=`.

    The warning message text is not asserted: older `pyiron_snippets` releases
    (exercised by the minimum-deps CI) emit the argument-deprecation message as
    an unformatted template, so only the `DeprecationWarning` category is
    reliable across versions.
    """
    rng = np.random.default_rng(0)
    blob_a = rng.uniform([0.0, 200.0], [0.2, 800.0], size=(20, 2))
    blob_b = rng.uniform([0.8, 200.0], [1.0, 800.0], size=(20, 2))
    df = pd.DataFrame(
        {
            "phase": ["A"] * 20 + ["B"] * 20,
            "c": np.concatenate([blob_a[:, 0], blob_b[:, 0]]),
            "T": np.concatenate([blob_a[:, 1], blob_b[:, 1]]),
            "stable": True,
        }
    )
    calls = []
    monkeypatch.setattr(
        plot_mod, "_plot_triplepoints", lambda df, ax=None, variables=None: calls.append(True)
    )
    with pytest.warns(DeprecationWarning):
        plot_phase_diagram(df, ax=ax, tielines=True, poly_method=Concave(drop_interior=False))
    assert calls == [True]  # tielines=True routed to triplepoints


# --- end-to-end on a real refined diagram ------------------------------------


@pytest.fixture(scope="module")
def eutectic_diagram():
    """Refined diagram for an hcp / fcc / liquid ideal-solution system
    (Basics.ipynb parameters) with a eutectic triple point where the three
    phases coexist."""
    fcc = ldp.IdealSolution(
        "fcc",
        ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB),
        ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * ldp.kB),
    )
    hcp = ldp.IdealSolution(
        "hcp",
        ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB),
        ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB),
    )
    liquid = ldp.IdealSolution(
        "liquid",
        ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * ldp.kB),
        ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * ldp.kB),
    )
    Ts = np.linspace(200.0, 1000.0, 25)
    return ldc.calc_phase_diagram([hcp, fcc, liquid], Ts, mu=50, refine=True)


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_real_diagram_marks_the_invariant(ax, eutectic_diagram, variables):
    """A real refined eutectic diagram drives both branches: an isothermal line
    spanning the coexisting concentrations in c-T, a marker at the same (mu, T)
    in mu-T."""
    triple = eutectic_diagram[eutectic_diagram["locus"] == Locus.TRIPLE]
    assert not triple.empty, "fixture must contain a triple point"
    T_t = triple["T"].mean()

    _plot_triplepoints(eutectic_diagram, ax=ax, variables=variables)
    # Compare component-wise: pytest.approx does not apply a tolerance inside a
    # nested [(x, y)] tuple, and the marker/line coordinates differ from their
    # column means at the last bit.
    if variables[0] == "c":
        assert _markers(ax) == []
        (y, xmin, xmax), = _hlines(ax)
        assert y == pytest.approx(T_t)
        assert (xmin, xmax) == pytest.approx((triple["c"].min(), triple["c"].max()))
    else:
        assert _hlines(ax) == []
        (mx, my), = _markers(ax)
        assert mx == pytest.approx(triple["mu"].mean())
        assert my == pytest.approx(T_t)
