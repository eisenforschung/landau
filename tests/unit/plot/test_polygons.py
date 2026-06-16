"""Unit tests for the public polygon-plotting plumbing in `landau.plot`.

These tests cover `get_polygons` and `plot_polygons` directly (issue #88).
They avoid going through `calc_phase_diagram` so each test is fast and
exercises one behaviour at a time.
"""
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import check_figures_equal, remove_ticks_and_titles

from shapely import Polygon as ShapelyPolygon
from shapely.ops import polylabel

from landau import plot as plot_mod
from landau.plot import (
    _add_inline_polygon_labels,
    _largest_inscribed_circle_center,
    _patch_outline_xy,
    _text_with_outline,
    get_phase_colors,
    get_polygons,
    plot_phase_diagram,
    plot_polygons,
)
from landau.poly import AbstractPolyMethod, BufferedSegments, Concave


# --- shared fixtures ---------------------------------------------------------


def _stable_df():
    """Two well-separated single-phase blobs in (c, T) space.

    Has a couple of unstable rows so the `query("stable")` filter has something
    to drop.  No `mu` column — `cluster_phase` calls `cluster(..., use_mu=False)`.
    """
    rng = np.random.default_rng(0)
    n_per_phase = 20
    blob_a = rng.uniform([0.0, 200.0], [0.2, 800.0], size=(n_per_phase, 2))
    blob_b = rng.uniform([0.8, 200.0], [1.0, 800.0], size=(n_per_phase, 2))
    df = pd.DataFrame(
        {
            "phase": ["A"] * n_per_phase + ["B"] * n_per_phase,
            "c": np.concatenate([blob_a[:, 0], blob_b[:, 0]]),
            "T": np.concatenate([blob_a[:, 1], blob_b[:, 1]]),
            "stable": True,
        }
    )
    unstable = pd.DataFrame(
        {
            "phase": ["A", "B"],
            "T": [500.0, 500.0],
            "c": [0.5, 0.6],
            "stable": [False, False],
        }
    )
    return pd.concat([df, unstable], ignore_index=True)


@dataclass
class _CapturingPoly(AbstractPolyMethod):
    """An `AbstractPolyMethod` that records the arguments passed to `apply`."""

    captured: list = field(default_factory=list)

    def apply(self, df, variables=["c", "T"]):
        self.captured.append({"df": df.copy(), "variables": list(variables)})
        return pd.Series([], dtype=object)

    def _make(self, pp, border, segment_label):  # pragma: no cover - not exercised
        return None


# --- get_polygons ------------------------------------------------------------


def test_get_polygons_filters_unstable_rows():
    cap = _CapturingPoly()
    get_polygons(_stable_df(), poly_method=cap)
    seen = cap.captured[0]["df"]
    assert seen["stable"].all()


def test_get_polygons_does_not_mutate_input():
    df = _stable_df()
    before = df.copy()
    get_polygons(df, poly_method=_CapturingPoly())
    pd.testing.assert_frame_equal(df, before)


def test_get_polygons_defaults_to_c_T():
    cap = _CapturingPoly()
    get_polygons(_stable_df(), poly_method=cap)
    assert cap.captured[0]["variables"] == ["c", "T"]


def test_get_polygons_forwards_variables():
    df = _stable_df().assign(mu=0.1)
    cap = _CapturingPoly()
    get_polygons(df, poly_method=cap, variables=["mu", "T"])
    assert cap.captured[0]["variables"] == ["mu", "T"]


def test_get_polygons_forwards_kwargs_to_handle_poly_method(monkeypatch):
    """`alpha` / `min_c_width` and the method name itself reach
    `handle_poly_method` unchanged."""
    seen = {}
    cap = _CapturingPoly()

    def fake_handle(poly_method, **kwargs):
        seen["poly_method"] = poly_method
        seen["kwargs"] = kwargs
        return cap

    monkeypatch.setattr(plot_mod.poly, "handle_poly_method", fake_handle)
    get_polygons(_stable_df(), poly_method="concave", alpha=0.37, min_c_width=0.05)

    assert seen["poly_method"] == "concave"
    assert seen["kwargs"] == {"alpha": 0.37, "min_c_width": 0.05}


def test_get_polygons_drops_failed_clusters_and_warns(monkeypatch):
    """Rows where `cluster_phase` produces phase_unit == -1 are dropped and
    a warning is emitted."""

    def fake_cluster_phase(df, **_):
        df = df.copy()
        df["phase_unit"] = [-1 if i == 0 else 0 for i in range(len(df))]
        df["phase_id"] = df["phase"].astype(str) + "_" + df["phase_unit"].astype(str)
        return df

    monkeypatch.setattr(plot_mod, "cluster_phase", fake_cluster_phase)

    cap = _CapturingPoly()
    with pytest.warns(UserWarning, match="Clustering of phase points failed"):
        get_polygons(_stable_df(), poly_method=cap)

    seen = cap.captured[0]["df"]
    assert (seen["phase_unit"] >= 0).all()


def test_get_polygons_returns_series_of_polygons():
    """End-to-end smoke test on a small synthetic frame: the returned Series
    is indexed by (phase, phase_unit) and its values are matplotlib Polygons.

    `drop_interior=False` is used because the synthetic frame has no `border`
    column; the default `drop_interior=True` would then return nothing.
    """
    result = get_polygons(_stable_df(), poly_method=Concave(drop_interior=False))
    assert isinstance(result, pd.Series)
    assert isinstance(result.index, pd.MultiIndex)
    assert list(result.index.names) == ["phase", "phase_unit"]
    assert len(result) > 0
    for poly in result:
        assert isinstance(poly, Polygon)


# --- plot_polygons -----------------------------------------------------------


def _square_polygon(xy=(0.0, 0.0), size=1.0):
    s = size
    x, y = xy
    return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])


def _polys_series(items):
    """Build a `pd.Series` of Polygons indexed by `items` (list of keys)."""
    polys = [_square_polygon((i, 0.0), 1.0) for i, _ in enumerate(items)]
    if items and isinstance(items[0], tuple):
        index = pd.MultiIndex.from_tuples(items, names=["phase", "phase_unit"])
    else:
        index = pd.Index(items, name="phase")
    return pd.Series(polys, index=index)


def test_plot_polygons_adds_one_patch_per_item():
    fig, ax = plt.subplots()
    polys = _polys_series([("A", 0), ("B", 0), ("A", 1)])
    color_map = {"A": "red", "B": "blue"}
    plot_polygons(polys, color_map, ax=ax)
    assert len(ax.patches) == 3
    plt.close(fig)


def test_plot_polygons_uses_provided_ax():
    fig_a, ax_a = plt.subplots()
    fig_b, ax_b = plt.subplots()
    plt.sca(ax_b)  # make ax_b the current axes
    polys = _polys_series([("A", 0)])
    plot_polygons(polys, {"A": "red"}, ax=ax_a)
    assert len(ax_a.patches) == 1
    assert len(ax_b.patches) == 0
    plt.close(fig_a)
    plt.close(fig_b)


def test_plot_polygons_defaults_to_plt_gca():
    fig, ax = plt.subplots()
    plt.sca(ax)
    polys = _polys_series([("A", 0)])
    plot_polygons(polys, {"A": "red"})
    assert len(ax.patches) == 1
    plt.close(fig)


def test_plot_polygons_label_and_color_for_scalar_key():
    fig, ax = plt.subplots()
    polys = _polys_series(["A"])
    plot_polygons(polys, {"A": "red"}, ax=ax)
    patch = ax.patches[0]
    assert patch.get_label() == "A"
    # set_color makes both facecolor and edgecolor the same — then edgecolor
    # gets overwritten to "k"; check the facecolor side.
    assert tuple(patch.get_facecolor()) == matplotlib.colors.to_rgba("red")
    plt.close(fig)


def test_plot_polygons_label_appends_apostrophes_for_phase_unit():
    fig, ax = plt.subplots()
    polys = _polys_series([("A", 0), ("A", 1), ("A", 2)])
    plot_polygons(polys, {"A": "red"}, ax=ax)
    labels = [p.get_label() for p in ax.patches]
    assert labels == ["A", "A'", "A''"]
    plt.close(fig)


def test_plot_polygons_edgecolor_is_black():
    fig, ax = plt.subplots()
    polys = _polys_series([("A", 0), ("B", 0)])
    plot_polygons(polys, {"A": "red", "B": "blue"}, ax=ax)
    for patch in ax.patches:
        assert tuple(patch.get_edgecolor()) == matplotlib.colors.to_rgba("k")
    plt.close(fig)


def test_plot_polygons_zorder_is_inverse_to_bbox_area():
    """Bigger polygons should sit behind smaller ones."""
    fig, ax = plt.subplots()
    big = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    small = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    polys = pd.Series(
        [big, small],
        index=pd.MultiIndex.from_tuples([("A", 0), ("B", 0)], names=["phase", "phase_unit"]),
    )
    plot_polygons(polys, {"A": "red", "B": "blue"}, ax=ax)
    z_big, z_small = (p.zorder for p in ax.patches)
    assert z_big < z_small
    plt.close(fig)


# --- inline-legend helpers ---------------------------------------------------


def test_text_with_outline_sets_white_stroke():
    fig, ax = plt.subplots()
    t = _text_with_outline(ax, 0.5, 0.5, "X", outline_width=4)
    effects = t.get_path_effects()
    assert len(effects) == 1
    stroke = effects[0]
    # The stroke is white and uses the requested linewidth.
    assert matplotlib.colors.to_rgba(stroke._gc["foreground"]) == matplotlib.colors.to_rgba("white")
    assert stroke._gc["linewidth"] == 4
    plt.close(fig)


def test_text_with_outline_forwards_kwargs():
    fig, ax = plt.subplots()
    t = _text_with_outline(ax, 1.0, 2.0, "Y", color="red", ha="center")
    assert t.get_text() == "Y"
    assert matplotlib.colors.to_rgba(t.get_color()) == matplotlib.colors.to_rgba("red")
    assert t.get_horizontalalignment() == "center"
    plt.close(fig)


def test_inscribed_circle_center_of_unit_square_is_centroid():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xy = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    cx, cy = _largest_inscribed_circle_center(xy, ax)
    assert cx == pytest.approx(0.5, abs=2e-3)
    assert cy == pytest.approx(0.5, abs=2e-3)
    plt.close(fig)


def test_inscribed_circle_center_handles_axis_anisotropy():
    """The label sits in the visually fat lobe, not where raw coordinates point.

    The polygon is an L on axes spanning c in [0, 1] and T in [0, 1000].  Its
    horizontal arm spans all of ``c`` but is only 0.4 K tall – a thin sliver on
    screen – while the vertical arm is the visually large region.  Without
    normalising by the axis ranges, the raw distance metric (where 0.4 in T
    dwarfs 1.0 in c) would place the centre in the sliver; normalising picks the
    vertical arm instead.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1000)
    xy = np.array(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.2, 0.4), (0.2, 1000.0), (0.0, 1000.0)]
    )
    cx, cy = _largest_inscribed_circle_center(xy, ax)
    # Vertical arm: above the sliver and within its narrow c-width.
    assert cy > 0.4
    assert cx < 0.2
    # A raw (un-normalised) pole of inaccessibility would land in the sliver.
    raw = polylabel(ShapelyPolygon(xy), tolerance=1e-3)
    assert raw.y < 0.4
    plt.close(fig)


def test_inscribed_circle_center_is_inside_polygon():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    xy = np.array([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)], dtype=float)
    center = _largest_inscribed_circle_center(xy, ax)
    assert ShapelyPolygon(xy).contains(_point(center))
    plt.close(fig)


def test_patch_outline_xy_open_stroke_uses_convex_hull():
    """An open BufferedSegments stroke labels at the region centre, not a
    self-intersecting concatenation of its disjoint border segments."""
    import shapely

    # Four disjoint border pieces roughly tracing a unit square, ordered so a
    # naive vertex concatenation would self-intersect (top before right).
    border = shapely.MultiLineString([
        [(0.0, 0.0), (1.0, 0.0)],   # bottom
        [(0.0, 1.0), (1.0, 1.0)],   # top
        [(1.0, 0.0), (1.0, 1.0)],   # right
        [(0.0, 0.0), (0.0, 1.0)],   # left
    ])
    patch = BufferedSegments()._to_mpl_polygon(border)
    outline = _patch_outline_xy(patch)
    # The outline is the convex hull of the border (the unit square), so its
    # pole of inaccessibility is the centre.
    hull = shapely.convex_hull(shapely.MultiPoint(outline))
    assert hull.equals(shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cx, cy = _largest_inscribed_circle_center(outline, ax)
    assert cx == pytest.approx(0.5, abs=2e-3)
    assert cy == pytest.approx(0.5, abs=2e-3)
    plt.close(fig)


def test_inscribed_circle_center_returns_none_for_degenerate():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Collinear points -> zero-area polygon.
    xy = np.array([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    assert _largest_inscribed_circle_center(xy, ax) is None
    plt.close(fig)


def test_add_inline_polygon_labels_one_text_per_polygon():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    polys = _polys_series([("A", 0), ("B", 0), ("A", 1)])
    _add_inline_polygon_labels(ax, polys)
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["A", "B", "A'"]
    # Labels are black for legibility on the white background.
    for t in ax.texts:
        assert matplotlib.colors.to_rgba(t.get_color()) == matplotlib.colors.to_rgba("black")
    plt.close(fig)


def _rect_patch(x0, x1, y0, y1):
    return Polygon(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), closed=True)


def _label_rects(ax, items):
    """Run `_add_inline_polygon_labels` on `(phase, x0, x1, y0, y1)` rectangles.

    Returns the created text artists, one per rectangle, in input order.
    """
    polys = pd.Series(
        [_rect_patch(x0, x1, y0, y1) for _, x0, x1, y0, y1 in items],
        index=pd.MultiIndex.from_tuples([(p, 0) for p, *_ in items], names=["phase", "phase_unit"]),
    )
    _add_inline_polygon_labels(ax, polys)
    return list(ax.texts)


def _label_rect(ax, phase, x0, x1, y0, y1):
    """Run `_add_inline_polygon_labels` on one rectangle and return its text artist."""
    (text,) = _label_rects(ax, [(phase, x0, x1, y0, y1)])
    return text


def _wide_axes():
    """Axes mimicking a c-T diagram: x in [0, 1], T in [0, 1000]."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1000)
    return fig, ax


def _extent_px(ax, text):
    return text.get_window_extent(plot_mod._get_renderer(ax.figure))


# Label placement is clamped in pixel space but stored in data coordinates, so
# round-tripping through the transform can wobble the bbox by float epsilon.
PX_TOL = 1e-6


def _assert_inside_axes(ax, bbox, vertical=True):
    axbb = ax.get_window_extent(plot_mod._get_renderer(ax.figure))
    assert axbb.x0 - PX_TOL <= bbox.x0 and bbox.x1 <= axbb.x1 + PX_TOL
    if vertical:
        assert axbb.y0 - PX_TOL <= bbox.y0 and bbox.y1 <= axbb.y1 + PX_TOL


def test_inline_label_horizontal_when_it_fits():
    fig, ax = _wide_axes()
    text = _label_rect(ax, "A", 0.1, 0.9, 100, 900)
    assert text.get_rotation() == 0
    x, y = text.get_position()
    assert x == pytest.approx(0.5, abs=2e-3)
    assert y == pytest.approx(500, abs=2)
    plt.close(fig)


def test_inline_label_rotates_in_tall_narrow_polygon():
    """A region narrower than the label but tall enough for it rotated keeps the pole anchor."""
    fig, ax = _wide_axes()
    text = _label_rect(ax, "ABCDEFGH", 0.47, 0.53, 50, 950)
    assert text.get_rotation() == 90
    x, y = text.get_position()
    assert x == pytest.approx(0.5, abs=2e-3)
    assert y == pytest.approx(500, abs=2)
    # The rotated box really is inside the polygon.
    bbox = _extent_px(ax, text)
    px0 = ax.transData.transform((0.47, 0.0))[0]
    px1 = ax.transData.transform((0.53, 0.0))[0]
    assert px0 < bbox.x0 and bbox.x1 < px1
    plt.close(fig)


def test_inline_label_moves_off_thin_polygon():
    """A line phase too thin for even a rotated label gets it placed beside, inside the axes."""
    fig, ax = _wide_axes()
    text = _label_rect(ax, "ABCDEFGH", 0.499, 0.501, 50, 950)
    assert text.get_rotation() == 90
    bbox = _extent_px(ax, text)
    px1 = ax.transData.transform((0.501, 0.0))[0]
    assert bbox.x0 > px1  # clear of the polygon, to its right
    _assert_inside_axes(ax, bbox)
    plt.close(fig)


def test_inline_label_off_polygon_flips_left_at_right_edge():
    """A terminal line phase at c=1 gets its label to the left, not outside the axes."""
    fig, ax = _wide_axes()
    text = _label_rect(ax, "ABCDEFGH", 0.998, 1.0, 50, 950)
    assert text.get_rotation() == 90
    bbox = _extent_px(ax, text)
    px0 = ax.transData.transform((0.998, 0.0))[0]
    assert bbox.x1 < px0  # clear of the polygon, to its left
    _assert_inside_axes(ax, bbox, vertical=False)
    plt.close(fig)


def test_inline_label_off_polygon_clamped_vertically():
    """A short line phase near the bottom edge keeps its label inside the axes."""
    fig, ax = _wide_axes()
    text = _label_rect(ax, "ABCDEFGH", 0.499, 0.501, 0, 60)
    assert text.get_rotation() == 90
    _assert_inside_axes(ax, _extent_px(ax, text))
    plt.close(fig)


def test_inline_offset_labels_spread_apart_vertically():
    """Coinciding line phases get offset labels fanned apart instead of stacked."""
    fig, ax = _wide_axes()
    t1, t2 = _label_rects(
        ax,
        [("ABCDEFGH", 0.499, 0.501, 50, 950), ("HGFEDCBA", 0.499, 0.501, 50, 950)],
    )
    assert t1.get_rotation() == 90 and t2.get_rotation() == 90
    b1, b2 = _extent_px(ax, t1), _extent_px(ax, t2)
    # Both polygons share the pole, so the boxes would coincide without the
    # overlap pass; now they must be vertically disjoint and stay in the axes.
    assert b1.y1 <= b2.y0 or b2.y1 <= b1.y0
    _assert_inside_axes(ax, b1)
    _assert_inside_axes(ax, b2)
    plt.close(fig)


def test_inline_offset_labels_far_apart_keep_pole_anchor():
    """Offset labels whose horizontal extents are clear of each other do not move."""
    fig, ax = _wide_axes()
    t1, t2 = _label_rects(
        ax,
        [("ABCDEFGH", 0.199, 0.201, 50, 950), ("HGFEDCBA", 0.799, 0.801, 50, 950)],
    )
    for t in (t1, t2):
        assert t.get_rotation() == 90
        assert t.get_position()[1] == pytest.approx(500, abs=2)
    plt.close(fig)


def test_plot_phase_diagram_inline_legend_default_labels_in_place():
    fig, ax = plt.subplots()
    plot_phase_diagram(_stable_df(), ax=ax, poly_method=Concave(drop_interior=False))
    assert ax.get_legend() is None
    labels = sorted(t.get_text() for t in ax.texts)
    assert labels == ["A", "B"]
    # Each label sits inside its polygon.
    for t in ax.texts:
        x, y = t.get_position()
        inside = any(p.get_path().contains_point((x, y)) for p in ax.patches)
        assert inside, f"label {t.get_text()!r} at {(x, y)} is outside every polygon"
    plt.close(fig)


def test_plot_phase_diagram_inline_legend_false_draws_legend():
    fig, ax = plt.subplots()
    plot_phase_diagram(
        _stable_df(), ax=ax, poly_method=Concave(drop_interior=False), inline_legend=False
    )
    assert ax.get_legend() is not None
    assert len(ax.texts) == 0
    plt.close(fig)


def _point(xy):
    from shapely import Point

    return Point(*xy)


# --- rendered-image equivalence tests ----------------------------------------
#
# `matplotlib.testing.decorators.check_figures_equal` renders both figures and
# compares the resulting images.  Used here to express visual properties that
# would otherwise need ad-hoc patch-attribute assertions and to validate the
# decomposition `plot_phase_diagram = get_polygons + plot_polygons + axis-setup`.


@check_figures_equal(extensions=["png"])
def test_plot_polygons_renders_independent_of_input_order(fig_test, fig_ref):
    """Because zorder is assigned from bbox area, presenting the polygons in
    different index orders should render to the same image."""

    def _build(ax):
        big = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        small = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        return big, small

    color_map = {"A": "red", "B": "blue"}

    ax_test = fig_test.subplots()
    big_t, small_t = _build(ax_test)
    plot_polygons(
        pd.Series(
            [big_t, small_t],
            index=pd.MultiIndex.from_tuples(
                [("A", 0), ("B", 0)], names=["phase", "phase_unit"]
            ),
        ),
        color_map,
        ax=ax_test,
    )
    ax_test.set_xlim(0, 10)
    ax_test.set_ylim(0, 10)

    ax_ref = fig_ref.subplots()
    big_r, small_r = _build(ax_ref)
    plot_polygons(
        pd.Series(
            [small_r, big_r],
            index=pd.MultiIndex.from_tuples(
                [("B", 0), ("A", 0)], names=["phase", "phase_unit"]
            ),
        ),
        color_map,
        ax=ax_ref,
    )
    ax_ref.set_xlim(0, 10)
    ax_ref.set_ylim(0, 10)


@check_figures_equal(extensions=["png"])
def test_plot_phase_diagram_matches_explicit_pipeline(fig_test, fig_ref):
    """`plot_phase_diagram(df, ax=ax)` should produce the same image as the
    explicit decomposition `get_polygons -> plot_polygons -> axis setup`.

    This guards against future refactors of either side drifting apart and
    documents what the public helper does in terms of the smaller helpers.
    """
    df = _stable_df()
    method = Concave(drop_interior=False)

    ax_test = fig_test.subplots()
    plot_phase_diagram(df, ax=ax_test, poly_method=method, inline_legend=False)

    df_stable = df.query("stable")
    color_map = get_phase_colors(df_stable.phase.unique())
    polys = get_polygons(df, poly_method=method)

    ax_ref = fig_ref.subplots()
    plot_polygons(polys, color_map, ax=ax_ref)
    ax_ref.set_xlim(0, 1)
    ax_ref.set_xlabel("$c$")
    ax_ref.set_ylim(df_stable["T"].min(), df_stable["T"].max())
    ax_ref.legend(ncols=2)
    ax_ref.set_ylabel("$T$ [K]")


@check_figures_equal(extensions=["png"])
def test_plot_polygons_ax_none_renders_identically_to_explicit_gca(fig_test, fig_ref):
    """ax=None (plt.gca() fallback) is pixel-identical to passing the same Axes explicitly.

    The existing functional test only checks patch count; this verifies the
    full rendered output is the same regardless of which code path resolves the
    target Axes.
    """
    color_map = {"A": "red", "B": "blue"}

    ax_ref = fig_ref.subplots()
    plot_polygons(_polys_series([("A", 0), ("B", 0)]), color_map, ax=ax_ref)
    remove_ticks_and_titles(fig_ref)

    ax_test = fig_test.subplots()
    plt.sca(ax_test)  # make ax_test the current axes so plt.gca() returns it
    plot_polygons(_polys_series([("A", 0), ("B", 0)]), color_map)  # ax=None
    remove_ticks_and_titles(fig_test)


@check_figures_equal(extensions=["png"])
def test_plot_polygons_scalar_key_identical_to_zero_rep_tuple(fig_test, fig_ref):
    """A Series keyed by plain string and one keyed by (phase, 0) render identically.

    Both index types resolve to rep=0 inside plot_polygons, so the label,
    facecolor, and edgecolor should produce the same pixels.
    """
    color_map = {"A": "red"}

    ax_ref = fig_ref.subplots()
    plot_polygons(_polys_series(["A"]), color_map, ax=ax_ref)
    remove_ticks_and_titles(fig_ref)

    ax_test = fig_test.subplots()
    plot_polygons(_polys_series([("A", 0)]), color_map, ax=ax_test)
    remove_ticks_and_titles(fig_test)


@check_figures_equal(extensions=["png"])
def test_get_and_plot_polygons_pipeline_deterministic(fig_test, fig_ref):
    """Two independent runs of get_polygons → plot_polygons on the same data
    produce pixel-identical figures (full pipeline determinism).
    """
    df = _stable_df()
    color_map = {"A": "red", "B": "blue"}

    ax_ref = fig_ref.subplots()
    plot_polygons(
        get_polygons(df.copy(), poly_method=Concave(drop_interior=False)),
        color_map,
        ax=ax_ref,
    )
    remove_ticks_and_titles(fig_ref)

    ax_test = fig_test.subplots()
    plot_polygons(
        get_polygons(df.copy(), poly_method=Concave(drop_interior=False)),
        color_map,
        ax=ax_test,
    )
    remove_ticks_and_titles(fig_test)
