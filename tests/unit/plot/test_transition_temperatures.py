"""Tests for transition-temperature annotations on 2d phase diagrams.

_annotate_transition_temperatures (plot_phase_diagram / plot_mu_phase_diagram's
transition_temperatures=True) labels every invariant with its temperature. Both
kinds are tagged in the dataframe -- Locus.TRIPLE for a three-phase invariant,
Locus.CONGRUENT for a point where two coexisting phases share a composition
(tagged by ClausiusClapeyronRefiner, tested in tests/unit/test_refine.py) -- so
these tests cover reading them back and, above all, where the labels land:
inside the axes and never across a phase boundary.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import shapely

from landau import plot as plot_mod
from landau.features import Locus
from landau.plot import (
    _annotate_transition_temperatures,
    _clear_label_center,
    _label_obstacles_px,
    _label_offsets,
    plot_mu_phase_diagram,
    plot_phase_diagram,
)
from landau.poly import Concave


_SIZE = (40.0, 12.0)  # a label's rendered (width, height) in pixels
_OFFSETS = None  # built per test from _SIZE; see _clear


def _clear(anchor, *, mode, x_weight, regions=(), obstacle=None, axes_box=None, size=_SIZE):
    """_clear_label_center with the offsets its caller would have built."""
    return _clear_label_center(
        anchor, size, _label_offsets(size, x_weight, step=4.0, max_offset=150.0),
        regions=list(regions), obstacle=obstacle, axes_box=axes_box, mode=mode,
    )


def _box(center, size=_SIZE):
    """Pixel box of a label of `size` centred on `center`."""
    (cx, cy), (w, h) = center, size
    return shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _phase_regions(ax):
    """The drawn phase polygons as pixel-space shapely polygons.

    Read back off the axes rather than from `get_polygons`, so the checks are
    against what a reader actually sees.
    """
    out = []
    for patch in ax.patches:
        region = plot_mod._shapely_polygon(ax.transData.transform(patch.get_xy()))
        if region is not None:
            out.append(region)
    return out


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


def _congruent_df(points):
    """Frame of Locus.CONGRUENT rows: one `(mu, T, c)` invariant per entry.

    Each carries the two coexisting phases at that shared composition, the way
    ClausiusClapeyronRefiner emits a tagged point.
    """
    rows = []
    for mu, T, c in points:
        for phase in ("S", "L"):
            rows.append({"mu": mu, "T": T, "c": c, "phase": phase,
                         "locus": Locus.CONGRUENT, "boundary_id": 0})
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


# --- _annotate_transition_temperatures ---------------------------------------


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_labels_every_triple_point(ax, triple_df, variables):
    _annotate_transition_temperatures(triple_df, ax=ax, variables=variables)
    assert sorted(t.get_text() for t in ax.texts) == ["300 K", "450 K"]


def test_cT_label_sits_above_the_triple_point(ax):
    """Anchored on the invariant's own composition -- the middle of the three,
    where the eutectic point is drawn -- not the midpoint of the isotherm it
    spans, which for an asymmetric invariant is somewhere else entirely."""
    df = pd.DataFrame(
        {
            "mu": [0.2] * 3,
            "T": [300.0] * 3,
            "c": [0.1, 0.3, 0.9],  # span midpoint 0.5, triple point 0.3
            "phase": ["A", "B", "C"],
            "locus": [Locus.TRIPLE] * 3,
        }
    )
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    label, = ax.texts
    x, y = label.get_position()
    assert x == pytest.approx(0.3)
    assert y > 300.0  # nudged above the isotherm
    assert label.get_ha() == "center"


def test_noop_without_locus_column(ax, triple_df):
    _annotate_transition_temperatures(triple_df.drop(columns="locus"), ax=ax)
    assert list(ax.texts) == []


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_labels_every_congruent_point(ax, variables):
    df = _congruent_df([(0.2, 320.0, 0.05), (0.6, 400.0, 0.95)])
    _annotate_transition_temperatures(df, ax=ax, variables=variables)
    assert sorted(t.get_text() for t in ax.texts) == ["320 K", "400 K"]
    assert list(ax.lines) == []  # the label alone, no marker


def test_congruent_label_anchors_on_the_shared_composition(ax):
    """One label per invariant, anchored on the composition the two phases
    share, not one per emitted row."""
    df = _congruent_df([(0.2, 320.0, 0.05)])
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    label, = ax.texts
    x, _y = label.get_position()
    assert x == pytest.approx(0.05)


def test_boundary_rows_are_not_labelled(ax):
    """Only tagged invariants are annotated; plain boundary rows are not."""
    df = _congruent_df([(0.2, 320.0, 0.05)]).assign(locus=Locus.BOUNDARY)
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    assert list(ax.texts) == [] and list(ax.lines) == []


# --- _clear_label_center -----------------------------------------------------

_AXES = shapely.box(0.0, 0.0, 400.0, 400.0)
_REGION = shapely.box(50.0, 50.0, 250.0, 250.0)  # one phase field


def test_field_mode_lands_wholly_inside_a_region():
    """Anchored on a boundary (as a congruent point is), the label moves into
    the phase field rather than straddling its edge."""
    center = _clear((150.0, 250.0), regions=[_REGION], obstacle=_REGION.exterior,
                    axes_box=_AXES, mode="field", x_weight=1.5)
    box = _box(center)
    assert _REGION.contains(box)
    assert not box.intersects(_REGION.exterior)


def test_negative_mode_stays_clear_of_every_region():
    """Same anchor, negative space: the label moves out of the field instead."""
    center = _clear((150.0, 250.0), regions=[_REGION], obstacle=_REGION.exterior,
                    axes_box=_AXES, mode="negative", x_weight=3.0)
    box = _box(center)
    assert not box.intersects(_REGION)
    assert center[1] > 250.0  # pushed out through the edge it was anchored on


def test_placement_stays_inside_the_axes():
    """A corner anchor gets pulled inwards; the box never leaves the axes."""
    center = _clear((0.0, 400.0), axes_box=_AXES, mode="free", x_weight=1.5)
    assert _AXES.contains(_box(center))


def test_placement_never_returns_the_anchor_itself():
    """The closest candidate still clears the labelled feature by half a label."""
    center = _clear((200.0, 200.0), axes_box=_AXES, mode="free", x_weight=1.5)
    assert abs(center[1] - 200.0) >= 6.0  # half the label height
    assert not _box(center).intersects(shapely.Point(200.0, 200.0))


def test_returns_none_when_nothing_fits():
    tiny = shapely.box(0.0, 0.0, 30.0, 30.0)  # narrower than the label
    assert _clear((15.0, 15.0), axes_box=tiny, mode="free", x_weight=1.5) is None


def test_label_falls_back_to_the_anchor_when_nothing_fits(ax, triple_df):
    """A label that cannot be placed anywhere clear is still drawn, pulled
    inside the axes at its anchor rather than dropped."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(300.0, 300.2)  # far shorter than a label is tall
    _annotate_transition_temperatures(triple_df, ax=ax, variables=["c", "T"])
    renderer = plot_mod._get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    boxes = _label_boxes(ax)
    assert boxes, "the label is kept, not dropped"
    for box in boxes:
        assert shapely.box(axbb.x0, axbb.y0, axbb.x1, axbb.y1).covers(box)


def test_labels_do_not_cover_each_other(ax):
    """Placed labels join the obstacles, so two invariants a few K apart get
    separate spots instead of one on top of the other."""
    df = _congruent_df([(0.2, 400.0, 0.5), (0.6, 403.0, 0.5)])
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    first, second = _label_boxes(ax)
    assert not first.intersects(second)


# --- _label_obstacles_px ------------------------------------------------------


def test_obstacles_cover_polygons_lines_markers_and_existing_labels(ax):
    """Everything a label has to keep off: the polygon outline, an isotherm
    drawn as a LineCollection, a single-point marker, and a label already on
    the axes."""
    poly = matplotlib.patches.Polygon([(0.1, 300.0), (0.9, 300.0), (0.9, 400.0)])
    ax.add_patch(poly)
    ax.hlines(350.0, 0.2, 0.8)
    ax.plot(0.5, 450.0, marker="o")
    ax.text(0.0, 480.0, "hcp")
    renderer = plot_mod._get_renderer(ax.figure)

    regions, obstacles = _label_obstacles_px(ax, [poly], renderer)
    assert len(regions) == 1
    geom = shapely.union_all(obstacles)
    assert geom.intersects(regions[0].exterior)                       # the outline
    assert geom.intersects(shapely.Point(ax.transData.transform((0.5, 350.0))))  # isotherm
    assert geom.intersects(shapely.Point(ax.transData.transform((0.5, 450.0))))  # marker
    assert geom.intersects(shapely.box(*ax.texts[0].get_window_extent(renderer).extents))


def test_obstacles_ignore_patches_the_caller_did_not_plot(ax):
    """Only the polygons handed in are phase regions; an unrelated patch on the
    axes is not one, and must not break the collection either."""
    ax.axvspan(0.2, 0.4)  # a Rectangle, whose get_xy() is a corner, not a ring
    renderer = plot_mod._get_renderer(ax.figure)
    regions, _obstacles = _label_obstacles_px(ax, [], renderer)
    assert regions == []


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
        regions = _phase_regions(ax)
        boxes = _label_boxes(ax)
        assert len(boxes) >= 2, "fixture carries a triple point and terminal congruent points"
        for box in boxes:
            assert axes_box.contains(box)
            for region in regions:
                assert not box.intersects(region.exterior)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_no_label_covers_another_with_the_legend_on(eutectic_diagram, variables):
    """The inline phase labels are drawn first, so the temperature labels have
    to keep off them too -- with legend=False, which the other end-to-end tests
    use, they are not there to collide with."""
    fig, ax = plt.subplots()
    try:
        plotter = plot_phase_diagram if variables[0] == "c" else plot_mu_phase_diagram
        plotter(eutectic_diagram, ax=ax, poly_method=Concave(drop_interior=False),
                transition_temperatures=True)
        renderer = plot_mod._get_renderer(fig)
        boxes = [shapely.box(*t.get_window_extent(renderer).extents) for t in ax.texts]
        assert len(boxes) >= 5  # three phases plus the invariants
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not a.intersects(b)
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
        regions = _phase_regions(ax)
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
        congruent = eutectic_diagram[eutectic_diagram["locus"] == Locus.CONGRUENT]
        assert not congruent.empty, "fixture must carry terminal congruent points"
        regions = _phase_regions(ax)
        for (_mu, T), _grp in congruent.groupby(["mu", "T"]):
            label, = [t for t in ax.texts if t.get_text() == f"{T:.0f} K"]
            box = shapely.box(*label.get_window_extent(renderer).extents)
            assert any(region.contains(box) for region in regions)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "triplepoints, transition_temperatures, marks, labels",
    [
        (None, False, False, False),   # neither asked for
        (None, True, True, True),      # unspecified: the marks follow the labels
        (True, False, True, False),    # marks alone, as before this feature
        (True, True, True, True),      # both asked for
        (False, True, False, True),    # marks refused: labels do not switch them back on
        (False, False, False, False),  # marks refused, nothing else asked for
    ],
    ids=["neither", "unspecified", "marks-only", "both", "marks-refused", "off"],
)
def test_triplepoint_marks_follow_the_caller(eutectic_diagram, triplepoints,
                                             transition_temperatures, marks, labels):
    """`triplepoints=None` is "the caller did not say", so the marks can follow
    the labels that annotate them. Saying it settles it: an explicit False keeps
    the marks off with the labels on, so no keyword switches on what another
    turned off -- and nothing has to warn about doing so."""
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_phase_diagram(
                eutectic_diagram, ax=ax, poly_method=Concave(drop_interior=False),
                triplepoints=triplepoints,
                transition_temperatures=transition_temperatures, legend=False,
            )
        # No arrangement of the two is surprising enough to warn about.
        assert not [w for w in caught if "triplepoints" in str(w.message)]
        isotherms = [seg for coll in ax.collections for seg in coll.get_segments()]
        assert bool(isotherms) is marks
        assert bool(_label_boxes(ax)) is labels
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
