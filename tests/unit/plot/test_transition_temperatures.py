"""Tests for transition-temperature annotations on 2d phase diagrams.

_annotate_transition_temperatures (plot_phase_diagram / plot_mu_phase_diagram's
transition_temperatures=True) labels every triple point with its temperature --
these are tagged in the dataframe (Locus.TRIPLE) -- and then does the same for
whatever congruent-transformation points _find_congruent_points turns up.
Congruent points are not tagged, so they are found heuristically: the
concentration gap between the two coexisting phases along a refined
boundary_id line shrinking below `tol`, either at a strict interior local
minimum (an isomorphous solidus/liquidus loop) or at the trace's own endpoint
provided that point's concentration is itself within `tol` of a domain edge
(c=0 or c=1) -- the trivial congruent case of a pure component's melting
point.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import shapely

from landau import plot as plot_mod
from landau.features import Locus
from landau.plot import (
    _annotate_transition_temperatures,
    _clear_label_center,
    _diagram_geometry_px,
    _find_congruent_points,
    plot_mu_phase_diagram,
    plot_phase_diagram,
)
from landau.poly import Concave


def _box(center, size=(40.0, 12.0)):
    """Pixel box of a label of `size` centred on `center`."""
    (cx, cy), (w, h) = center, size
    return shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _label_boxes(ax):
    """Pixel boxes of the temperature labels drawn on `ax`."""
    renderer = plot_mod._get_renderer(ax.figure)
    return [
        shapely.box(*t.get_window_extent(renderer).extents)
        for t in ax.texts
        if t.get_text().endswith(" K")
    ]


@pytest.fixture
def ax():
    """A fresh axes, its limits covering the synthetic fixtures below.

    Placement is axes-aware -- a label is only ever put where it fits inside the
    axes -- so the anchors have to be in view for these tests to exercise
    anything but the out-of-view fallback.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(250.0, 500.0)
    yield ax
    plt.close(fig)


def _boundary_df(gaps, Ts, mus=None, phases=("S", "L"), boundary_id=0, centers=None):
    """A two-phase refined boundary_id line with a given concentration gap per T.

    `gaps[i]` is the concentration gap between the two coexisting phases at
    `Ts[i]` (`mus[i]` if given, else 0, 1, 2, ...), centered on `centers[i]`
    (0.5 -- mid-composition -- for every point if not given).
    """
    if mus is None:
        mus = [float(i) for i in range(len(Ts))]
    if centers is None:
        centers = [0.5] * len(Ts)
    lo, hi = phases
    rows = []
    for mu, T, gap, center in zip(mus, Ts, gaps, centers):
        c_lo, c_hi = center - gap / 2, center + gap / 2
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


def test_edge_minimum_away_from_terminal_excluded():
    """A shrinking gap at the trace's end is not enough on its own: it also has
    to sit near a domain edge (c=0 or c=1), or the trace just stopped there for
    some other reason (e.g. a triple point away from either terminal)."""
    df = _boundary_df(gaps=[0.30, 0.20, 0.10, 0.02], Ts=[300.0, 310.0, 320.0, 330.0])  # centered on c=0.5
    assert _find_congruent_points(df, tol=0.05) == []


def test_terminal_congruent_point_included():
    """The trivial congruent case: a solidus/liquidus trace running into a pure
    component's melting point at c=0 has its gap and its own concentration
    both vanish together at that endpoint."""
    df = _boundary_df(
        gaps=[0.02, 0.20, 0.30], Ts=[300.0, 320.0, 340.0], centers=[0.01, 0.15, 0.30],
    )
    out = _find_congruent_points(df, tol=0.05)
    assert len(out) == 1
    mu, T, c = out[0]
    assert (T, c) == pytest.approx((300.0, 0.01))


def test_endpoint_gap_shrinking_but_off_terminal_excluded():
    """Same shrinking-gap-at-the-endpoint shape as the terminal case, but the
    composition itself never approaches 0 or 1 -- not a terminal melting
    point, so it must not be flagged."""
    df = _boundary_df(
        gaps=[0.02, 0.20, 0.30], Ts=[300.0, 320.0, 340.0], centers=[0.5, 0.5, 0.5],
    )
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


def test_cT_label_sits_above_the_line_midpoint(ax, triple_df):
    _annotate_transition_temperatures(triple_df, ax=ax, variables=["c", "T"])
    by_text = {t.get_text(): t for t in ax.texts}
    x, y = by_text["300 K"].get_position()
    assert x == pytest.approx(0.5)  # midpoint of this invariant's c=0.1..0.9
    assert y > 300.0  # nudged above the isotherm
    assert by_text["300 K"].get_ha() == "center"


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


# --- _clear_label_center -----------------------------------------------------

_AXES = shapely.box(0.0, 0.0, 400.0, 400.0)
_REGION = shapely.box(50.0, 50.0, 250.0, 250.0)  # one phase field


def test_field_mode_lands_wholly_inside_a_region():
    """Anchored on a boundary (as a congruent point is), the label moves into
    the phase field rather than straddling its edge."""
    center = _clear_label_center(
        (150.0, 250.0), (40.0, 12.0), regions=[_REGION], obstacle=_REGION.exterior,
        axes_box=_AXES, mode="field", x_weight=1.5,
    )
    box = _box(center)
    assert _REGION.contains(box)
    assert not box.intersects(_REGION.exterior)


def test_negative_mode_stays_clear_of_every_region():
    """Same anchor, negative space: the label moves out of the field instead."""
    center = _clear_label_center(
        (150.0, 250.0), (40.0, 12.0), regions=[_REGION], obstacle=_REGION.exterior,
        axes_box=_AXES, mode="negative", x_weight=3.0,
    )
    box = _box(center)
    assert not box.intersects(_REGION)
    assert center[1] > 250.0  # pushed out through the edge it was anchored on


def test_placement_stays_inside_the_axes():
    """A corner anchor gets pulled inwards; the box never leaves the axes."""
    center = _clear_label_center(
        (0.0, 400.0), (40.0, 12.0), regions=[], obstacle=None,
        axes_box=_AXES, mode="free", x_weight=1.5,
    )
    assert _AXES.contains(_box(center))


def test_placement_never_returns_the_anchor_itself():
    """The closest candidate still clears the labelled feature by half a label."""
    center = _clear_label_center(
        (200.0, 200.0), (40.0, 12.0), regions=[], obstacle=None,
        axes_box=_AXES, mode="free", x_weight=1.5,
    )
    assert abs(center[1] - 200.0) >= 6.0  # half the label height
    assert not _box(center).intersects(shapely.Point(200.0, 200.0))


def test_returns_none_when_nothing_fits():
    tiny = shapely.box(0.0, 0.0, 30.0, 30.0)  # narrower than the label
    assert _clear_label_center(
        (15.0, 15.0), (40.0, 12.0), regions=[], obstacle=None,
        axes_box=tiny, mode="free", x_weight=1.5,
    ) is None


# --- end-to-end on a real refined diagram ------------------------------------


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_labels_stay_inside_the_axes_and_off_every_phase_boundary(eutectic_diagram, variables):
    """The placement guarantee, checked against a real diagram's own geometry."""
    fig, ax = plt.subplots()
    try:
        plotter = plot_phase_diagram if variables[0] == "c" else plot_mu_phase_diagram
        plotter(
            eutectic_diagram, ax=ax, poly_method=Concave(drop_interior=False),
            transition_temperatures=True, legend=False,
        )
        renderer = plot_mod._get_renderer(fig)
        axbb = ax.get_window_extent(renderer)
        axes_box = shapely.box(axbb.x0, axbb.y0, axbb.x1, axbb.y1)
        regions, _obstacles = _diagram_geometry_px(ax)
        boxes = _label_boxes(ax)
        assert len(boxes) >= 2, "fixture carries a triple point and terminal congruent points"
        for box in boxes:
            assert axes_box.contains(box)
            for region in regions:
                assert not box.intersects(region.exterior)
    finally:
        plt.close(fig)


def test_triple_label_sits_in_the_two_phase_negative_space(eutectic_diagram):
    """In c-T the invariant's label goes above or below its isotherm, in the
    negative space -- not into either single-phase field."""
    fig, ax = plt.subplots()
    try:
        plot_phase_diagram(
            eutectic_diagram, ax=ax, poly_method=Concave(drop_interior=False),
            transition_temperatures=True, legend=False,
        )
        renderer = plot_mod._get_renderer(fig)
        T_t = eutectic_diagram[eutectic_diagram["locus"] == Locus.TRIPLE]["T"].mean()
        label, = [t for t in ax.texts if t.get_text() == f"{T_t:.0f} K"]
        box = shapely.box(*label.get_window_extent(renderer).extents)
        regions, _obstacles = _diagram_geometry_px(ax)
        assert regions, "the diagram must have drawn phase polygons"
        for region in regions:
            assert not box.intersects(region)
    finally:
        plt.close(fig)


def test_congruent_label_sits_inside_a_phase_field(eutectic_diagram):
    """A congruent point sits on the edge between two fields; its label goes
    into one of them."""
    fig, ax = plt.subplots()
    try:
        plot_phase_diagram(
            eutectic_diagram, ax=ax, poly_method=Concave(drop_interior=False),
            transition_temperatures=True, legend=False,
        )
        renderer = plot_mod._get_renderer(fig)
        congruent = _find_congruent_points(eutectic_diagram)
        assert congruent, "fixture must carry terminal congruent points"
        regions, _obstacles = _diagram_geometry_px(ax)
        for _mu, T, _c in congruent:
            label, = [t for t in ax.texts if t.get_text() == f"{T:.0f} K"]
            box = shapely.box(*label.get_window_extent(renderer).extents)
            assert any(region.contains(box) for region in regions)
    finally:
        plt.close(fig)


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
