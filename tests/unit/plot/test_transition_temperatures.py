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

from landau.features import Locus
from landau.plot import plot_mu_phase_diagram, plot_phase_diagram
from landau.plot.labels import (
    _LABEL_PAD,
    _LABEL_REACH,
    _TemperatureLabel,
    _annotate_transition_temperatures,
    _label_candidates,
    _get_renderer,
    _label_obstacles_px,
    _place_temperature_labels,
    _shapely_polygon,
)
from landau.poly import Concave


_SIZE = (40.0, 12.0)  # a label's rendered (width, height) in pixels
_AXES = shapely.box(0.0, 0.0, 400.0, 400.0)
_REGION = shapely.box(50.0, 50.0, 250.0, 250.0)  # one phase field


def _label(anchor, T=300.0, *, modes=("free",), x_weight=1.5, span=None, size=_SIZE):
    """A label to place, anchored on an invariant at `anchor` (pixels)."""
    return _TemperatureLabel(T=T, anchor=anchor, size=size, modes=modes, x_weight=x_weight,
                             span=(anchor[0], anchor[0]) if span is None else span)


def _place(labels, *, regions=(), obstacle=None, axes_box=_AXES):
    """Place `labels` and return their centres, in the order given."""
    _place_temperature_labels(list(labels), list(regions), obstacle, axes_box)
    return [label.center for label in labels]


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
        region = _shapely_polygon(ax.transData.transform(patch.get_xy()))
        if region is not None:
            out.append(region)
    return out


def _label_boxes(ax):
    """Pixel boxes of the temperature labels drawn on `ax`."""
    renderer = _get_renderer(ax.figure)
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


# --- _place_temperature_labels -----------------------------------------------


def test_field_mode_lands_wholly_inside_a_region():
    """Anchored on a boundary (as a congruent point is), the label moves into
    the phase field rather than straddling its edge."""
    label = _label((150.0, 250.0), modes=("field", "negative", "free"))
    center, = _place([label], regions=[_REGION], obstacle=_REGION.exterior.buffer(_LABEL_PAD))
    box = _box(center)
    assert _REGION.contains(box)
    assert not box.intersects(_REGION.exterior)


def test_negative_mode_stays_clear_of_every_region():
    """Same anchor, negative space: the label moves out of the field instead."""
    label = _label((150.0, 250.0), modes=("negative", "field", "free"), x_weight=2.0)
    center, = _place([label], regions=[_REGION], obstacle=_REGION.exterior.buffer(_LABEL_PAD))
    box = _box(center)
    assert not box.intersects(_REGION)
    assert center[1] > 250.0  # pushed out through the edge it was anchored on


def test_placement_stays_inside_the_axes():
    """A corner anchor gets pulled inwards; the box never leaves the axes."""
    center, = _place([_label((0.0, 400.0))])
    assert _AXES.contains(_box(center))


def test_placement_never_returns_the_anchor_itself():
    """The closest candidate still clears the labelled feature by half a label."""
    center, = _place([_label((200.0, 200.0))])
    assert abs(center[1] - 200.0) >= 6.0  # half the label height
    assert not _box(center).intersects(shapely.Point(200.0, 200.0))


def test_wide_label_on_the_axes_edge_still_has_candidates():
    """A terminal melting point sits on the axes edge, and a four-digit label
    is wider than the reach in heights: the grid still offers spots inside."""
    wide = (44.0, 12.0)  # wider than 2 * 12 px of horizontal reach
    label = _label((400.0, 200.0), size=wide)  # anchored on the right edge of _AXES
    cands = _label_candidates(label, _AXES)
    assert len(cands) > 0
    for cx, cy in cands:
        assert _AXES.contains(_box((cx, cy), wide).buffer(_LABEL_PAD))


def test_no_candidate_when_the_axes_cannot_hold_the_label():
    tiny = shapely.box(0.0, 0.0, 30.0, 30.0)  # narrower than the label
    center, = _place([_label((15.0, 15.0))], axes_box=tiny)
    assert center is None


def test_labels_keep_temperature_order_when_crowded():
    """Three invariants half a label apart in y: their labels cannot all sit
    next to their own isotherm, but however they are stacked the hotter one is
    never drawn under the cooler one."""
    labels = [_label((200.0, 200.0 + 6.0 * i), T=700.0 + 50.0 * i) for i in range(3)]
    centers = _place(labels)
    ys = [cy for _cx, cy in centers]
    assert ys == sorted(ys)
    reach = _LABEL_REACH * _SIZE[1] + _SIZE[1] / 2 + _LABEL_PAD
    for label, (cx, cy) in zip(labels, centers):
        assert abs(cy - label.anchor[1]) <= reach + 1e-9


def test_label_keeps_off_the_wrong_side_of_a_neighbouring_invariant():
    """A 700 K label whose nearest spot would reach across the 720 K isotherm
    just above it goes below its own isotherm instead, and the 720 K label
    above its own -- each on the side that keeps the temperatures in order,
    with neither label anywhere near the other's spot."""
    cool = _label((200.0, 200.0), T=700.0, span=(100.0, 300.0))
    hot = _label((200.0, 210.0), T=720.0, span=(100.0, 300.0))
    (_cx, y_cool), (_cx, y_hot) = _place([cool, hot])
    assert y_cool < 200.0
    assert y_hot > 210.0


def test_crossing_is_ignored_far_along_the_isotherm():
    """The order constraint against another invariant only applies where the
    label is horizontally near it: an isotherm far to the right does not push
    a label off its preferred side."""
    near = _label((200.0, 200.0), T=700.0)
    far = _label((350.0, 210.0), T=720.0, span=(330.0, 370.0))
    (_cx, y_near), _ = _place([near, far])
    assert y_near > 200.0  # kept the nearest spot, above its own isotherm


def test_crowded_label_stays_within_reach_of_its_anchor():
    """When every candidate overlaps something, the label overplots next to its
    anchor rather than drifting to a clear spot far away."""
    everything = shapely.box(-10.0, -10.0, 410.0, 410.0)
    center, = _place([_label((200.0, 200.0))], obstacle=everything)
    assert abs(center[1] - 200.0) <= _SIZE[1] / 2 + _LABEL_PAD + 1e-9
    assert center[0] == pytest.approx(200.0)


def test_duplicate_invariants_share_one_label(ax):
    """Two invariants at the same rounded temperature within a label of each
    other -- a eutectic next to a terminal melting point -- get one label."""
    df = pd.concat([
        _congruent_df([(0.2, 300.0, 0.50)]),
        _congruent_df([(0.4, 300.2, 0.51)]),
    ], ignore_index=True)
    _annotate_transition_temperatures(df, ax=ax, variables=["c", "T"])
    assert [t.get_text() for t in ax.texts] == ["300 K"]


def test_label_falls_back_to_the_anchor_when_nothing_fits(ax, triple_df):
    """A label that cannot be placed anywhere inside the axes is still drawn,
    pulled in at its anchor rather than dropped."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(300.0, 300.2)  # the 450 K invariant is far out of view
    _annotate_transition_temperatures(triple_df, ax=ax, variables=["c", "T"])
    renderer = _get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    boxes = _label_boxes(ax)
    assert len(boxes) == 2, "both labels are kept, not dropped"
    for box in boxes:
        assert shapely.box(axbb.x0, axbb.y0, axbb.x1, axbb.y1).covers(box)


def test_labels_do_not_cover_each_other(ax):
    """Two invariants a few K apart have room on either side of their
    isotherms, so they take separate spots instead of one on top of the
    other."""
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
    renderer = _get_renderer(ax.figure)

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
    renderer = _get_renderer(ax.figure)
    regions, _obstacles = _label_obstacles_px(ax, [], renderer)
    assert regions == []


# --- end-to-end on a real refined diagram ------------------------------------


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
def test_labels_stay_inside_the_axes_and_off_every_phase_boundary(eutectic_diagram, variables):
    """On a diagram with room around every invariant, no label covers a phase
    boundary, checked against the diagram's own geometry.

    Drawn with the default hull: `Concave(drop_interior=False)` cuts slots
    into the mu-T fields on this coarse grid, and next to those artefact
    outlines no spot within a label's reach is clear."""
    fig, ax = plt.subplots()
    try:
        plotter = plot_phase_diagram if variables[0] == "c" else plot_mu_phase_diagram
        plotter(eutectic_diagram, ax=ax, transition_temperatures=True, legend=False)
        renderer = _get_renderer(fig)
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
        renderer = _get_renderer(fig)
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
        renderer = _get_renderer(fig)
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
        renderer = _get_renderer(fig)
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
