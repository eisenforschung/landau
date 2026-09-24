"""Text placement for the phase-diagram plots.

Everything that decides *where* a label goes lives here, separate from the
functions in :mod:`landau.plot` that decide *what* is drawn: the inline
phase-field labels and transition-temperature annotations of the 2d diagrams,
the top-spine and side labels of the 1d cuts, and the curve labels of the
excess-free-energy plot, together with the geometry helpers they share.
Placement works in display pixels throughout, because a rendered label has a
fixed pixel size while the two data axes have unrelated scales.
"""

from dataclasses import dataclass, field

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import to_rgba
from shapely.ops import polylabel

from ..features import Locus


def _text_with_outline(ax, x, y, s, *, outline_width=3, **kwargs):
    """Draw text with a solid white outline so it stays legible over any fill.

    A small reusable wrapper around :meth:`matplotlib.axes.Axes.text` that
    strokes the glyphs with a white outline (via matplotlib path effects)
    instead of drawing an opaque box behind them, keeping a label readable on
    top of coloured regions or tielines.  Extra keyword arguments are forwarded
    to ``ax.text``.

    Returns the created :class:`matplotlib.text.Text`.
    """
    kwargs.setdefault(
        "path_effects",
        [patheffects.withStroke(linewidth=outline_width, foreground="white")],
    )
    return ax.text(x, y, s, **kwargs)


def _get_renderer(fig):
    """Return a renderer for *fig*, drawing the canvas first so text can be measured.

    ``Text.get_window_extent`` needs a renderer to report the rendered pixel size of a
    label.  ``Figure.canvas.get_renderer`` exists on the Agg backend; other backends
    expose it via the private ``Figure._get_renderer`` (matplotlib >= 3.6).
    """
    fig.canvas.draw()
    if hasattr(fig.canvas, "get_renderer"):
        return fig.canvas.get_renderer()
    return fig._get_renderer()


def _bold_math(label: str) -> str:
    """Wrap mathtext segments of ``label`` in ``\\mathbf`` so subscripts render bold.

    matplotlib's ``fontweight="bold"`` only affects the regular-text portions of a
    string; content inside ``$...$`` keeps the regular math weight. Wrapping each
    math segment in ``\\mathbf{...}`` bolds it to match while leaving arbitrary
    LaTeX inside renderable. A malformed string (odd number of ``$``) is returned
    unchanged.
    """
    parts = label.split("$")
    if len(parts) % 2 == 0:  # unbalanced '$' -> leave untouched
        return label
    for i in range(1, len(parts), 2):
        if parts[i]:
            parts[i] = r"\mathbf{" + parts[i] + "}"
    return "$".join(parts)


def _shapely_polygon(coords):
    """Valid shapely polygon from an (N, 2) coordinate array, or ``None``.

    A self-intersecting outline (as the TSP-based poly methods can produce) is
    repaired with :func:`shapely.make_valid`; if the repair splits it into
    several pieces, the largest one is kept.  Degenerate (empty or zero-area)
    input gives ``None``.
    """
    poly = shapely.Polygon(coords)
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
        if isinstance(poly, shapely.MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    if not isinstance(poly, shapely.Polygon) or poly.is_empty or poly.area == 0:
        return None
    return poly


def _spread_labels(centers, heights, lo, hi, gap=0.0):
    """Nudge label centers apart so their vertical extents never overlap.

    Each label *i* occupies ``[center - heights[i]/2, center + heights[i]/2]``.  The
    returned centers keep the input order's stacking, sit within ``[lo, hi]`` where the
    stack fits, and move as little as possible from the requested ``centers``.  Works in
    whatever 1-d coordinate the caller passes (display pixels are convenient because a
    rendered text height is constant there regardless of the data scale).

    Args:
        centers: Desired center coordinate of each label.
        heights: Full extent of each label in the same units as ``centers``.
        lo, hi: Lower / upper bounds the stack should stay within.
        gap: Extra clearance to keep between adjacent labels.

    Returns:
        List of adjusted centers, one per input label, in the input order.
    """
    n = len(centers)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: centers[i])
    adj = list(centers)
    # Upward pass: push each label up just enough to clear the one below it.
    prev_top = lo
    for i in order:
        half = heights[i] / 2
        c = max(centers[i], prev_top + half)
        adj[i] = c
        prev_top = c + half + gap
    # Downward pass: pull labels down to respect the top bound while staying separated.
    next_bot = hi
    for i in reversed(order):
        half = heights[i] / 2
        c = min(adj[i], next_bot - half)
        adj[i] = c
        next_bot = c - half - gap
    return adj


def _group_overlapping_intervals(intervals, gap=0.0):
    """Group 1-d intervals that overlap, transitively, closer than ``gap``.

    Returns lists of indices into ``intervals``; within a group every interval
    overlaps (or comes within ``gap`` of) another one, so the members have to be
    laid out together along the other axis.
    """
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    groups, hi = [], None
    for i in order:
        lo_i, hi_i = intervals[i]
        if hi is None or lo_i > hi + gap:
            groups.append([])
            hi = hi_i
        else:
            hi = max(hi, hi_i)
        groups[-1].append(i)
    return groups


def _phase_visible_in_band(phi, lo, hi):
    """Whether a line with potentials ``phi`` (in scan order) shows inside ``[lo, hi]``.

    Visible if any sampled point falls within the band, or if two consecutive points
    straddle it (the connecting segment crosses the whole window).  Used to drop the
    label of a phase whose line an applied ``ylim`` pushes entirely out of view.
    """
    phi = np.asarray(phi, dtype=float)
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        return False
    if np.any((phi >= lo) & (phi <= hi)):
        return True
    below, above = phi < lo, phi > hi
    crosses = (below[:-1] & above[1:]) | (above[:-1] & below[1:])
    return bool(np.any(crosses))


def _curve_obstacles(ax):
    """Everything drawn on *ax* in data coordinates, as one pixel-space geometry.

    Curves and segment collections (a phase curve, a triple point's isotherm)
    become :class:`shapely.LineString`\\ s; scatter dots and lone markers
    (line-phase markers, hull vertices, a triple point in mu-T) become
    :class:`shapely.Point`\\ s buffered by their marker radius.  All in display
    coordinates, so a label's pixel bounding box can be tested against them
    directly.  The horizontal reference line (a blended-transform ``axhline``)
    is skipped so labels are not pushed off the zero line.  Returns the unioned
    geometry, or ``None`` when nothing is drawn yet.
    """
    geoms = []
    dpi = ax.figure.dpi
    for line in ax.lines:
        if line.get_transform() is not ax.transData:  # skip refline / blended artists
            continue
        disp = line.get_transform().transform(line.get_xydata())
        disp = disp[np.isfinite(disp).all(axis=1)]
        if len(disp) >= 2:
            geoms.append(shapely.LineString(disp))
        elif len(disp) == 1 and line.get_marker() not in ("", "None", None):
            # A lone marker (a triple point in mu-T) has no line to trace.
            radius = max(line.get_markersize() / 2.0 * dpi / 72.0, 1.0)
            geoms.append(shapely.Point(disp[0]).buffer(radius))
    for coll in ax.collections:
        # Segment collections (a triple point's isotherm, from ax.hlines).
        for seg in getattr(coll, "get_segments", list)():
            seg = coll.get_transform().transform(seg)
            if len(seg) >= 2:
                geoms.append(shapely.LineString(seg))
    for coll in ax.collections:
        if not hasattr(coll, "get_sizes"):  # not a scatter; handled above
            continue
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        if offsets.size == 0:
            continue
        disp = coll.get_offset_transform().transform(offsets)
        sizes = coll.get_sizes()
        for i, (px, py) in enumerate(disp):
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            s = sizes[i % len(sizes)] if len(sizes) else 36.0
            radius = max((np.sqrt(s) / 2.0) * dpi / 72.0, 1.0)  # points^2 -> pixel radius
            geoms.append(shapely.Point(px, py).buffer(radius))
    return shapely.union_all(geoms) if geoms else None


def _largest_inscribed_circle_center(polygon_xy, ax):
    """Centre of the largest circle inscribable in a polygon, in data units.

    Uses shapely's pole of inaccessibility (:func:`shapely.ops.polylabel`),
    which lies inside even concave or crescent-shaped regions.  Phase-diagram
    axes are strongly anisotropic (``c`` spans ~1, ``T`` spans hundreds of
    kelvin), so the coordinates are normalised by the axis data-ranges before
    the search and mapped back afterwards; this yields the visually – rather
    than numerically – largest circle.

    Returns ``(x, y)`` in data coordinates, or ``None`` for a degenerate
    polygon.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = (x1 - x0) or 1.0
    sy = (y1 - y0) or 1.0
    coords = np.asarray(polygon_xy, dtype=float)
    poly = _shapely_polygon(np.column_stack([coords[:, 0] / sx, coords[:, 1] / sy]))
    if poly is None:
        return None
    point = polylabel(poly, tolerance=1e-3)
    return point.x * sx, point.y * sy


def _label_fits(poly_px, text, renderer):
    """Whether ``text``'s rendered bounding box lies inside a pixel-space polygon."""
    return poly_px.contains(shapely.box(*text.get_window_extent(renderer).extents))


def _add_inline_polygon_labels(ax, polys):
    """Label each phase polygon in place instead of drawing a legend box.

    Every polygon is annotated with its phase name (with trailing apostrophes
    for repeated stability regions, matching :func:`plot_polygons`), with a
    white outline.  The text is black: the polygon fill already carries the
    phase colour, and black with a white stroke stays legible even over the
    pale pastel fills.

    Placement tries three positions in turn, keeping the first whose rendered
    bounding box fits inside the polygon:

    1. horizontal at the centre of the largest inscribed circle,
    2. rotated by 90° at the same point — for tall, narrow regions,
    3. still rotated, but moved horizontally off the polygon — for line phases
       too thin to hold any label.  The label sits just right of the polygon
       unless that would leave the axes (a terminal line phase at the right
       edge), in which case it sits to the left; it is clamped into the axes
       both ways, vertically too.

    Offset labels are no longer anchored inside their own polygon, so two close
    line phases can land on top of each other; a final pass fans labels with
    overlapping horizontal extents apart vertically (via :func:`_spread_labels`,
    within the axes), so each one slides along its line instead of covering its
    neighbour.

    Args:
        ax: matplotlib Axes the polygons were drawn on.
        polys: Series of matplotlib Polygons indexed as in :func:`get_polygons`.
    """
    renderer = _get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    pad = 0.004 * axbb.width  # clearance between an offset label box and its polygon
    moved = []  # (text, cx, cy, width, height) of off-polygon labels, in pixels
    for key, p in polys.items():
        if isinstance(key, tuple):
            phase, rep = key
        else:
            phase, rep = key, 0
        center = _largest_inscribed_circle_center(p.get_xy(), ax)
        if center is None:
            continue
        text = _text_with_outline(
            ax, center[0], center[1], phase + "'" * rep,
            ha="center", va="center", fontsize="small", fontweight="bold",
            color="black",
        )
        poly_px = _shapely_polygon(ax.transData.transform(p.get_xy()))
        if poly_px is None or _label_fits(poly_px, text, renderer):
            continue
        text.set_rotation(90)
        if _label_fits(poly_px, text, renderer):
            continue
        # Too thin even for a rotated label (a line phase): move it beside the
        # polygon, keeping the rotation.
        bbox = text.get_window_extent(renderer)
        half_w = bbox.width / 2
        minx, _, maxx, _ = poly_px.bounds
        cx = maxx + pad + half_w
        if cx + half_w > axbb.x1:
            cx = minx - pad - half_w
        cx = min(max(cx, axbb.x0 + half_w), axbb.x1 - half_w)
        cy = ax.transData.transform(center)[1]
        moved.append((text, cx, cy, bbox.width, bbox.height))

    # Vertical overlap pass: spread offset labels whose horizontal extents
    # collide.  _spread_labels also clamps every stack — singletons included —
    # into the axes vertically.
    inv = ax.transData.inverted()
    intervals = [(cx - w / 2, cx + w / 2) for _, cx, _, w, _ in moved]
    for group in _group_overlapping_intervals(intervals, gap=pad):
        spread = _spread_labels(
            [moved[i][2] for i in group],
            [moved[i][4] for i in group],
            axbb.y0, axbb.y1, gap=pad,
        )
        for i, cy in zip(group, spread):
            text, cx = moved[i][0], moved[i][1]
            text.set_position(inv.transform((cx, cy)))


_LABEL_PAD = 3.0  # px clearance kept between a label box and any drawn feature


def _label_obstacles_px(ax, polys, renderer):
    """What a temperature label has to stay off, in display pixels.

    ``regions`` are the phase polygons as drawn -- taken from the same
    :func:`get_polygons` result the caller plotted, rather than read back off
    ``ax.patches``, which would also pick up any unrelated patch the caller put
    there. ``obstacles`` are the lines it must not cover: every polygon
    outline except where it runs along the edge of the sampled window or the
    axes frame (a phase field cut off there has an edge, but no boundary a
    reader could mistake a label for covering), whatever else is drawn in data
    coordinates (a triple point's isotherm and marker, via
    :func:`_curve_obstacles`) and the labels already placed on the axes. Pixel
    space is the natural frame here: a label's rendered size is fixed in
    pixels while the two data axes have unrelated scales.
    """
    regions = []
    for patch in polys:
        region = _shapely_polygon(ax.transData.transform(patch.get_xy()))
        if region is not None:
            regions.append(region)
    edges = [shapely.box(*ax.get_window_extent(renderer).extents).exterior]
    if regions:
        edges.append(shapely.box(*shapely.union_all(regions).bounds).exterior)
    frame = shapely.union_all(edges).buffer(_LABEL_PAD)
    obstacles = []
    for region in regions:
        outline = region.exterior.difference(frame)
        if not outline.is_empty:
            obstacles.append(outline)
    drawn = _curve_obstacles(ax)
    if drawn is not None:
        obstacles.append(drawn)
    obstacles.extend(shapely.box(*t.get_window_extent(renderer).extents) for t in ax.texts)
    return regions, obstacles


def _label_offsets(size, x_weight, step, max_offset):
    """Candidate displacements from an anchor, nearest first.

    Horizontal movement is charged ``x_weight`` times vertical, so a label
    slides along whichever axis it is meant to have room on before drifting
    sideways. The closest candidate already clears the labelled feature by half
    a label plus the pad, so the anchor itself is never returned.
    """
    lo = size[1] / 2 + _LABEL_PAD
    n = int(max_offset / step) + 1
    offsets = [(sx * i * step, sy * (lo + j * step))
               for i in range(n) for j in range(n)
               for sx in ((1, -1) if i else (1,)) for sy in (1, -1)]
    offsets.sort(key=lambda d: float(np.hypot(d[0] * x_weight, d[1])))
    return offsets


# Placement cost weights. Every term is measured in label heights, so they
# compare like for like against the distance term, whose largest value is the
# reach of the candidate set: 2 heights, costing 4. A line crossing the box
# covers about a third of it and so costs as much as that whole reach, as does
# reaching half a label past another invariant's temperature; either is avoided
# whenever any candidate within reach avoids it, and accepted otherwise.
_LABEL_COST = {
    "distance": 2.0,   # per label height moved off the anchor (x charged x_weight times)
    "overlap": 12.0,   # per padded label-box area covered by a feature or another label
    "mode": 1.0,       # per step down the label's list of preferred spaces
    "crossing": 8.0,   # per label height the box reaches past another invariant's temperature
    "swap": 8.0,       # per label height two labels' centres are out of temperature order
}
_LABEL_REACH = 2.0     # candidates extend this many label heights from the anchor
_LABEL_SWEEPS = 5      # repair passes over the greedy placement


@dataclass
class _TemperatureLabel:
    """One transition-temperature label while it is being placed.

    Everything is in display pixels. ``anchor`` is the invariant itself (its
    ``y`` is the temperature level every other label is ordered against),
    ``span`` the horizontal extent of the feature it labels -- an isotherm's
    two ends in c-T, the point itself in mu-T -- and ``modes`` the label's
    preferred spaces, best first, as :func:`_box_modes` names them.
    ``candidates`` are the centres it may take, ``choice`` the index of the one
    it has, ``None`` while unplaced or when nothing fits inside the axes.
    """

    T: float
    anchor: tuple[float, float]
    size: tuple[float, float]
    modes: tuple[str, ...]
    x_weight: float
    span: tuple[float, float]
    candidates: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    static_cost: np.ndarray = field(default_factory=lambda: np.empty(0))
    choice: int | None = None

    @property
    def center(self):
        return None if self.choice is None else (float(self.candidates[self.choice, 0]),
                                                 float(self.candidates[self.choice, 1]))


def _label_candidates(label, axes_box):
    """Candidate centres for ``label``: the :func:`_label_offsets` grid out to
    :data:`_LABEL_REACH` label heights, keeping the padded box inside ``axes_box``.

    The grid reaches at least half a label width plus a step sideways, whatever
    the reach in heights, so a label anchored on the axes edge -- a terminal
    melting point at c=0 or c=1 -- still has candidates that fit inside.
    """
    w, h = label.size
    step = max(h / 4.0, 1.0)
    reach = max(_LABEL_REACH * h, w / 2 + _LABEL_PAD + step)
    offsets = np.asarray(_label_offsets(label.size, label.x_weight, step, reach))
    centers = np.asarray(label.anchor, dtype=float) + offsets
    hw, hh = w / 2 + _LABEL_PAD, h / 2 + _LABEL_PAD
    x0, y0, x1, y1 = axes_box.bounds
    inside = ((centers[:, 0] - hw >= x0) & (centers[:, 0] + hw <= x1)
              & (centers[:, 1] - hh >= y0) & (centers[:, 1] + hh <= y1))
    return centers[inside]


def _box_modes(boxes, regions):
    """Which space each of ``boxes`` sits in, given the phase-field ``regions``:

    ``"field"``
        wholly inside one phase polygon. A congruent point sits on the edge
        between two single-phase fields, so either of them will do.
    ``"negative"``
        clear of every phase polygon, i.e. in the two-phase negative space --
        where a triple point's isotherm runs, with a two-phase field above it
        and another below.
    ``"free"``
        neither: straddling a boundary.
    """
    if not regions:
        return np.full(len(boxes), "negative")
    inside = np.any([shapely.contains(r, boxes) for r in regions], axis=0)
    touching = np.any([shapely.intersects(r, boxes) for r in regions], axis=0)
    return np.where(inside, "field", np.where(touching, "free", "negative"))


def _static_costs(label, regions, obstacle, levels):
    """Cost of each candidate of ``label`` from what does not move: its
    distance from the anchor, how much of its padded box ``obstacle`` covers,
    the space it lands in, and how far it reaches past another invariant's
    temperature.

    ``levels`` are the other invariants as ``(T, span, y)``; a candidate whose
    box comes within a label width of one's ``span`` pays for every pixel the
    box reaches past that temperature on the wrong side, beyond the pad, so a
    label may touch a neighbouring isotherm but not sit across it. A crossing is
    penalised whether or not the neighbour's label is anywhere near, which is
    what keeps the order of the labels honest even when they do not interact.
    """
    c = label.candidates
    if len(c) == 0:
        return np.empty(0)
    w, h = label.size
    hw, hh = w / 2 + _LABEL_PAD, h / 2 + _LABEL_PAD
    weights = _LABEL_COST
    cost = weights["distance"] * np.hypot((c[:, 0] - label.anchor[0]) * label.x_weight,
                                          c[:, 1] - label.anchor[1]) / h
    if obstacle is not None:
        padded = shapely.box(c[:, 0] - hw, c[:, 1] - hh, c[:, 0] + hw, c[:, 1] + hh)
        cost += weights["overlap"] * shapely.area(shapely.intersection(padded, obstacle)) / (4 * hw * hh)
    # The space a label sits in is judged by the text itself, without the pad:
    # next to the edge of the sampled window the pad pokes out of every field.
    boxes = shapely.box(c[:, 0] - w / 2, c[:, 1] - h / 2, c[:, 0] + w / 2, c[:, 1] + h / 2)
    rank = {m: i for i, m in enumerate(label.modes)}
    cost += weights["mode"] * np.array([rank.get(m, len(label.modes)) for m in _box_modes(boxes, regions)])
    for T, (sx0, sx1), y in levels:
        if T == label.T:
            continue
        near = (c[:, 0] + 1.5 * w > sx0) & (c[:, 0] - 1.5 * w < sx1)
        if label.T > T:
            depth = (y - _LABEL_PAD) - (c[:, 1] - h / 2)
        else:
            depth = (c[:, 1] + h / 2) - (y + _LABEL_PAD)
        cost += weights["crossing"] * np.where(near, np.maximum(depth, 0.0), 0.0) / h
    return cost


def _dynamic_costs(label, placed):
    """Cost of each candidate of ``label`` from the labels in ``placed``: the
    part of its padded box they cover, and, for one it shares horizontal
    extent with, how far its centre sits on the wrong side of theirs in
    temperature order -- the 800 K label under the 700 K one -- scaled by that
    shared extent, so labels far apart in x do not constrain each other."""
    c = label.candidates
    cost = np.zeros(len(c))
    w, h = label.size
    hw, hh = w / 2 + _LABEL_PAD, h / 2 + _LABEL_PAD
    weights = _LABEL_COST
    for other in placed:
        (ox, oy), (ow, oh) = other.center, other.size
        ix = np.minimum(c[:, 0] + hw, ox + ow / 2) - np.maximum(c[:, 0] - hw, ox - ow / 2)
        iy = np.minimum(c[:, 1] + hh, oy + oh / 2) - np.maximum(c[:, 1] - hh, oy - oh / 2)
        cost += weights["overlap"] * np.clip(ix, 0.0, None) * np.clip(iy, 0.0, None) / (4 * hw * hh)
        if other.T == label.T:
            continue
        shared = np.clip(np.minimum(c[:, 0] + w / 2, ox + ow / 2)
                         - np.maximum(c[:, 0] - w / 2, ox - ow / 2), 0.0, None) / min(w, ow)
        depth = oy - c[:, 1] if label.T > other.T else c[:, 1] - oy
        cost += weights["swap"] * shared * np.maximum(depth, 0.0) / h
    return cost


def _place_temperature_labels(labels, regions, obstacle, axes_box):
    """Choose a centre for every label in ``labels``, in place.

    Each label scores its candidates by :func:`_static_costs` plus
    :func:`_dynamic_costs` and takes the cheapest. Labels are placed hottest
    first, so each one only ever sees the higher temperatures already fixed
    above it, then re-placed one at a time with the others held still until
    none moves (at most :data:`_LABEL_SWEEPS` passes): the greedy pass alone
    lets an early label push a later one into a worse spot than the two could
    share. No candidate is ever rejected outright except for leaving the axes,
    so a crowded invariant is overplotted near its anchor rather than labelled
    somewhere clear but far away.
    """
    levels = [(label.T, label.span, label.anchor[1]) for label in labels]
    for label in labels:
        label.candidates = _label_candidates(label, axes_box)
        label.static_cost = _static_costs(label, regions, obstacle, levels)
        label.choice = None
    order = sorted(range(len(labels)), key=lambda i: -labels[i].T)
    placed = []
    for i in order:
        label = labels[i]
        if len(label.candidates) == 0:
            continue
        label.choice = int(np.argmin(label.static_cost + _dynamic_costs(label, placed)))
        placed.append(label)
    for _ in range(_LABEL_SWEEPS):
        moved = False
        for label in placed:
            total = label.static_cost + _dynamic_costs(label, [o for o in placed if o is not label])
            best = int(np.argmin(total))
            if total[best] < total[label.choice] - 1e-9:
                label.choice, moved = best, True
        if not moved:
            break


def _annotate_transition_temperatures(df, polys=(), ax=None, variables=None):
    """Label the temperature of every transition invariant on a 2d phase diagram.

    Both kinds of invariant are tagged in the ``locus`` column of a refined
    :func:`~landau.calculate.calc_phase_diagram` frame, so this only reads them
    back: :attr:`~landau.features.Locus.TRIPLE` for a three-phase invariant
    (see :func:`_plot_triplepoints`) and
    :attr:`~landau.features.Locus.CONGRUENT` for a point where two coexisting
    phases share a composition (see
    :meth:`~landau.refine.ClausiusClapeyronRefiner._tag_features`). A frame
    with neither -- an unrefined one, say -- draws nothing. Invariants whose
    labels would read the same and whose anchors lie within one label of each
    other -- a eutectic and a terminal melting point at the same rounded
    temperature -- share one label at their mean anchor.

    Every label is anchored on its invariant -- a triple point at the middle
    of its three compositions, the phase that melts or decomposes into the
    outer two and where the eutectic or peritectic point is drawn; a congruent
    point at the shared composition -- and placed by
    :func:`_place_temperature_labels`: the cheapest of the candidate spots
    within two label heights of the anchor, weighing the distance moved
    against covering a phase boundary, a drawn curve or another label, leaving
    the space it prefers, and reaching past another invariant's temperature or
    out of temperature order with another label. Each kind prefers the space
    its feature lives in:

    * a triple point's label goes into the two-phase negative space above or
      below its isotherm, which exists only in c-T: in mu-T the phase fields
      tile the plane, so neither kind looks for it there,
    * a congruent point's label goes inside one of the two phase fields meeting
      at it.

    Nothing near the anchor is ever ruled out, so a crowded invariant gets an
    overplotted label next to itself rather than a clear one far away; only a
    label with no room at all inside the axes is pulled in at its anchor.

    Args:
        df (pandas.DataFrame):
            Phase-diagram frame carrying a ``locus`` column. Unrefined frames
            have nothing to label.
        polys (pandas.Series or iterable, optional):
            The phase polygons as drawn, from :func:`get_polygons`; a label is
            kept off their outlines and, for a congruent point, inside one of
            them.
        ax (matplotlib.axes.Axes, optional):
            The axis to plot on.
        variables (list[str], optional):
            The ``[x, y]`` axis variables; defaults to ``["c", "T"]``.
    """
    if variables is None:
        variables = ["c", "T"]
    if ax is None:
        ax = plt.gca()
    if "locus" not in df.columns:
        return

    # The two-phase negative space is a c-T notion; in mu-T the phase fields
    # tile the plane, so looking for it there only pays for a doomed scan.
    if variables[0] == "c":
        triple_modes, congruent_modes = ("negative", "field", "free"), ("field", "negative", "free")
    else:
        triple_modes = congruent_modes = ("field", "free")

    # (x, T, x-extent, modes, x_weight) per invariant, in data coordinates.
    entries = []
    triple = df[df["locus"] == Locus.TRIPLE]
    if variables[0] == "c":
        for (_mu, T), grp in triple.groupby(["mu", "T"], sort=False)[["c"]]:
            entries.append((grp["c"].median(), T, (grp["c"].min(), grp["c"].max()), triple_modes, 2.0))
    elif variables[0] == "mu":
        for (mu, T), _grp in triple.groupby(["mu", "T"], sort=False):
            entries.append((mu, T, (mu, mu), triple_modes, 2.0))
    congruent = df[df["locus"] == Locus.CONGRUENT]
    for (mu, T), grp in congruent.groupby(["mu", "T"], sort=False)[["c"]]:
        # The two phases meet here, so their concentrations agree to within the
        # refiner's tolerance; the mean is the composition of the invariant.
        x = grp["c"].mean() if variables[0] == "c" else mu
        entries.append((x, T, (x, x), congruent_modes, 1.5))
    if not entries:
        return

    renderer = _get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    axes_box = shapely.box(axbb.x0, axbb.y0, axbb.x1, axbb.y1)
    regions, obstacles = _label_obstacles_px(ax, polys, renderer)
    obstacle = shapely.union_all([o.buffer(_LABEL_PAD) for o in obstacles]) if obstacles else None

    texts, labels = [], []
    for x, T, (x0, x1), modes, x_weight in entries:
        text = _text_with_outline(
            ax, x, T, f"{T:.0f} K", ha="center", va="center", fontsize="small", zorder=11,
        )
        bbox = text.get_window_extent(renderer)
        ax_px, ay_px = ax.transData.transform((x, T))
        span = tuple(sorted((ax.transData.transform((x0, T))[0], ax.transData.transform((x1, T))[0])))
        label = _TemperatureLabel(T=T, anchor=(float(ax_px), float(ay_px)), size=(bbox.width, bbox.height),
                                  modes=modes, x_weight=x_weight, span=span)
        # Coalesce with an earlier label that reads the same and sits within
        # one label of this one: keep that one, at the mean anchor.
        for other_text, other in zip(texts, labels):
            if (other_text.get_text() == text.get_text()
                    and abs(other.anchor[0] - label.anchor[0]) <= other.size[0]
                    and abs(other.anchor[1] - label.anchor[1]) <= other.size[1]):
                other.anchor = ((other.anchor[0] + label.anchor[0]) / 2, (other.anchor[1] + label.anchor[1]) / 2)
                other.span = (min(other.span[0], span[0]), max(other.span[1], span[1]))
                text.remove()
                break
        else:
            texts.append(text)
            labels.append(label)

    _place_temperature_labels(labels, regions, obstacle, axes_box)

    inv = ax.transData.inverted()
    for text, label in zip(texts, labels):
        center = label.center
        if center is None:
            # Nothing fits inside the axes: keep the label at its anchor,
            # pulled inside with the same clearance a placed one would keep.
            half_w, half_h = label.size[0] / 2 + _LABEL_PAD, label.size[1] / 2 + _LABEL_PAD
            center = (min(max(label.anchor[0], axbb.x0 + half_w), axbb.x1 - half_w),
                      min(max(label.anchor[1], axbb.y0 + half_h), axbb.y1 - half_h))
        text.set_position(inv.transform(center))


def _place_side_labels(ax, df, scan_col, phase_colors, top_texts=()):
    """Label every phase at the end of its line, reserving room adaptively.

    A phase is labelled at its right-hand line end by default.  The horizontal space
    reserved for the stack is derived from the widest *rendered* label (measured in
    pixels), not from any string-length heuristic, so the axis is widened by exactly as
    much as the labels need.  Within a stack the labels are spread vertically so they do
    not overlap, clamped to the current y-limits.

    If the current y-limit would clip a label off the top (its right-end value lies above
    the visible window), that phase is instead labelled at its left-hand line end on a
    mirrored left-hand stack, which is reserved and laid out by the same rules.  A phase
    whose line the y-limit pushes entirely out of view is not labelled at all.

    ``top_texts`` are the bold top stable-phase labels; both stacks are capped just below
    their rendered bottom so they never intrude into that band.
    """
    fig = ax.figure
    x_min, x_max = df[scan_col].min(), df[scan_col].max()
    span0 = x_max - x_min

    # Settle autoscaled limits (and obtain a renderer) before deciding which side each
    # label belongs to or measuring any text.
    renderer = _get_renderer(fig)
    lo_d, hi_d = sorted(ax.get_ylim())
    axbb = ax.get_window_extent(renderer)

    # Ceiling for both stacks: just below the bold top-label band so they don't collide.
    hi_px = axbb.y1
    if len(top_texts):
        top_bottom = min(t.get_window_extent(renderer).y0 for t in top_texts)
        hi_px = max(min(hi_px, top_bottom - 0.01 * axbb.height), axbb.y0)

    # Split phases: those visible at the right end label on the right; those whose
    # right end is above the window label at their left end instead.  A phase whose
    # whole line is pushed out of view by the y-limit gets no label at all.
    right, left = [], []
    for phase, group in df.groupby("phase"):
        g = group.sort_values(scan_col)
        phi = g["phi"].to_numpy()
        if not _phase_visible_in_band(phi, lo_d, hi_d):
            continue
        right_y = g["phi"].iloc[-1]
        if right_y > hi_d:
            left.append((phase, g["phi"].iloc[0]))
        else:
            right.append((phase, right_y))

    def make_texts(entries, ha):
        return [
            ax.text(
                x_min, y, phase, transform=ax.transData,
                ha=ha, va="center", fontsize="small",
                color=phase_colors.get(phase, "black"), clip_on=True,
            )
            for phase, y in entries
        ]

    right_texts = make_texts(right, "left")
    left_texts = make_texts(left, "right")

    def measure(texts):
        if not texts:
            return 0.0, []
        ext = [t.get_window_extent(renderer) for t in texts]
        return max(e.width for e in ext) / axbb.width, [e.height for e in ext]

    gap_frac, margin_frac = 0.02, 0.01
    f_right, h_right = measure(right_texts)
    f_left, h_left = measure(left_texts)
    reserve_r = (gap_frac + f_right + margin_frac) if right_texts else 0.0
    reserve_l = (gap_frac + f_left + margin_frac) if left_texts else 0.0

    # Widen the axis so the line data occupies the middle (1 - reserve_l - reserve_r) of
    # it and each stack sits in its reserved strip.
    denom = max(1.0 - reserve_l - reserve_r, 0.2)
    total_span = span0 / denom
    x0 = x_min - reserve_l * total_span
    ax.set_xlim(x0, x0 + total_span)

    def place(texts, entries, heights, anchor_frac):
        if not texts:
            return
        anchor_x = x0 + anchor_frac * total_span
        # Spread the true (unclamped) line-terminal pixels: _spread_labels bounds the
        # result to the axes box itself, so the labels stay in the window while keeping
        # the terminals' vertical order.  Pre-clamping would tie targets that fall
        # outside the window and collapse that order.
        target_px = [ax.transData.transform((anchor_x, y))[1] for _, y in entries]
        placed_px = _spread_labels(target_px, heights, axbb.y0, hi_px)
        inv = ax.transData.inverted()
        for t, py in zip(texts, placed_px):
            y_d = inv.transform((axbb.x0, py))[1]
            t.set_position((anchor_x, y_d))

    place(right_texts, right, h_right, 1.0 - margin_frac - f_right)
    place(left_texts, left, h_left, f_left + margin_frac)


def _place_transition_labels(ax, positions, labels, *, side, **text_kw):
    """Label transition lines with vertical text, spread to avoid overlaps.

    Each transition at ``positions[i]`` is annotated with ``labels[i]`` as rotated
    (vertical) text just inside the bottom of the axes.  Each label is first offset
    to one side of its line, then the whole row is fanned apart along x by the same
    :func:`_spread_labels` routine that stacks the side labels vertically -- this
    removes every mutual overlap while preserving the labels' left-to-right order.
    A final best-effort pass shifts any label that still straddles a dotted
    transition line clear of it, but only within the slack its neighbours leave, so
    it never re-introduces an overlap; lines packed closer than a label is wide
    cannot all be avoided and are left as they fall.

    ``side`` ('left' or 'right') biases the initial offset of each label relative
    to its own line.

    Returns the created :class:`~matplotlib.text.Text` artists.
    """
    if len(positions) == 0:
        return []
    order = np.argsort(positions)
    positions = [positions[i] for i in order]
    labels = [labels[i] for i in order]

    renderer = _get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    # y in axes fraction (blended transform) so a zoomed ylim can't fling the text
    # out of view; only the x position is spread.
    xform = ax.get_xaxis_transform()
    texts = [
        _text_with_outline(
            ax, x, 0.02, s, transform=xform,
            rotation="vertical", ha="center", va="bottom", fontsize="small", zorder=100, **text_kw,
        )
        for x, s in zip(positions, labels)
    ]
    widths = [t.get_window_extent(renderer).width for t in texts]

    # Spreading is done in display pixels (a rendered width is constant there), then
    # mapped back to the data x the blended transform expects.
    lines_px = [ax.transData.transform((x, 0.0))[0] for x in positions]
    pad = 0.004 * axbb.width  # clearance kept between a label box and a line
    half = [w / 2 for w in widths]
    offset = -1 if side == "left" else 1

    # Offset each label to one side of its line, then fan them apart (order- and
    # bound-preserving) so no two boxes overlap.
    centers = [lines_px[i] + offset * (half[i] + pad) for i in range(len(texts))]
    centers = _spread_labels(centers, widths, axbb.x0, axbb.x1, gap=pad)

    # Best-effort: shift any label that still covers a transition line off it,
    # clamped to the slack between its neighbours (and the axes) so the no-overlap
    # guarantee from the spread survives.
    def covered(c, i):
        return sum(c - half[i] - pad < L < c + half[i] + pad for L in lines_px)

    for i, c in enumerate(centers):
        lo = axbb.x0 + half[i] if i == 0 else centers[i - 1] + half[i - 1] + half[i] + pad
        hi = axbb.x1 - half[i] if i == len(centers) - 1 else centers[i + 1] - half[i + 1] - half[i] - pad
        if lo > hi or covered(c, i) == 0:
            continue
        # Candidate slots sit the box just left or right of each line; clamp to the
        # neighbour slack and keep the reachable one that covers the fewest lines.
        slots = [L + s * (half[i] + pad) for L in lines_px for s in (-1, 1)]
        cands = [c] + [min(max(s, lo), hi) for s in slots]
        centers[i] = min(cands, key=lambda cc: (covered(cc, i), abs(cc - c)))

    inv = ax.transData.inverted()
    for t, c in zip(texts, centers):
        t.set_position((inv.transform((c, 0.0))[0], 0.02))

    return texts


def _add_1d_phase_legend(ax, df, scan_col, top_labels=True, side_labels=True, ylim=None):
    """Annotate a 1d phase diagram with inline phase labels.

    top_labels
        Ticks on the top spine mark each transition boundary (positions where
        ``border == True``) and the stable phase name is placed near the top of
        the axis, centered between adjacent boundaries.

    side_labels
        The default seaborn legend is removed and, when unstable lines are
        present, every phase is annotated by name at the right end of its final
        line segment inside the axis.

    Args:
        ax: matplotlib Axes with a seaborn lineplot already rendered.
        df: DataFrame with 'phase', 'stable', ``scan_col``, 'phi', and
            (if available) 'border' columns.
        scan_col: Scan-axis column ('mu' or 'T').
        top_labels: If True, add the top-spine ticks and stable-phase labels.
        side_labels: If True, remove the default seaborn legend and add the
            right-end side labels.
        ylim: If given, applied via ``ax.set_ylim`` before the labels are placed,
            so the side labels are clamped to (and spread within) this window.
            A scalar is treated as ``(None, ylim)`` — an upper bound only.
    """
    if ylim is not None:
        if np.isscalar(ylim):
            ylim = (None, ylim)
        ax.set_ylim(ylim)

    # Extract phase → color from the seaborn legend (used by both label sets).
    phase_colors = {}
    legend = ax.get_legend()
    if legend is not None:
        white = to_rgba("w")
        gray2 = to_rgba(".2")
        for handle, text in zip(legend.legend_handles, legend.texts):
            try:
                c = handle.get_color()
            except AttributeError:
                continue
            if to_rgba(c) in (white, gray2):
                continue
            phase_colors[text.get_text()] = c
        # The side labels replace the default seaborn legend.
        if side_labels:
            legend.remove()

    # Store so tests (or user code) can still access the color map.
    ax._landau_phase_colors = phase_colors

    top_texts = []
    if top_labels and "border" in df.columns:
        # Interior transition positions only.
        x_min = df[scan_col].min()
        x_max = df[scan_col].max()
        transitions = sorted(
            t for t in df.loc[df["border"], scan_col].unique()
            if x_min < t < x_max
        )
        boundaries = [x_min] + transitions + [x_max]

        # Top-spine ticks at transition boundaries (no tick labels; just marks).
        ax2 = ax.secondary_xaxis("top")
        ax2.set_ticks(transitions)
        ax2.set_xticklabels([])
        ax2.tick_params(direction="in", length=6)

        # Phase-name labels near the top of the axis, centered between boundaries.
        # get_xaxis_transform(): x in data coords, y in axes [0,1] fraction.
        xform = ax.get_xaxis_transform()
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            mid = (lo + hi) / 2
            # Strict inequalities exclude the border rows themselves, which can have
            # two phases marked stable simultaneously (the transition point).
            mask = (df[scan_col] > lo) & (df[scan_col] < hi) & df["stable"]
            stable_phases = df.loc[mask, "phase"].unique()
            if len(stable_phases) != 1:
                # A stable window narrower than the grid spacing has no sample
                # strictly inside its two borders; the phase marked stable on
                # both enclosing rows is the one stable throughout.
                at_lo = set(df.loc[(df[scan_col] == lo) & df["stable"], "phase"])
                at_hi = set(df.loc[(df[scan_col] == hi) & df["stable"], "phase"])
                stable_phases = sorted(at_lo & at_hi)
            if len(stable_phases) != 1:
                raise RuntimeError(
                    f"expected exactly one stable phase in [{lo}, {hi}], "
                    f"got {list(stable_phases)}"
                )
            phase = stable_phases[0]
            # White outline keeps the bold label legible on top of tielines.
            top_texts.append(_text_with_outline(
                ax, mid, 0.97, _bold_math(phase),
                transform=xform,
                ha="center", va="top", fontsize="small", fontweight="bold",
                color=phase_colors.get(phase, "black"),
            ))

    if side_labels and not df["stable"].all():
        _place_side_labels(ax, df, scan_col, phase_colors, top_texts)


def _add_inline_curve_labels(ax, entries):
    """Label curves/points just off the data line instead of via a legend box.

    Each entry ``(label, x, y, color, side)`` is anchored at ``(x, y)`` on the
    curve (or dot) and drawn one label-height clear of it rather than on top of it:
    ``side="above"`` lifts the label into the open (convex) side of a free-energy
    curve, ``side="below"`` drops it under a lower-hull line-phase dot.  Labels are
    white-outlined in the phase colour with mathtext subscripts bolded by
    :func:`_bold_math`, then nudged apart along ``y`` in display pixels by
    :func:`_spread_labels` so they never overlap, while ``x`` stays anchored to the
    curve.  A final pass tests each label's pixel box (via shapely) against the
    drawn curves, scatter markers (:func:`_curve_obstacles`) and the
    already-placed labels, pushing it further in its ``side`` direction until it no
    longer overlaps any of them or it reaches the axes edge.  Mirrors the inline
    labelling used by the 2d (:func:`_add_inline_polygon_labels`) and 1d
    (:func:`_place_side_labels`) plots.

    Args:
        ax: matplotlib Axes the curves were drawn on.
        entries: iterable of ``(label, x, y, color, side)`` anchors, one per phase,
            with ``side`` one of ``"above"`` / ``"below"``.

    Returns the created :class:`~matplotlib.text.Text` artists.
    """
    entries = list(entries)
    if not entries:
        return []
    renderer = _get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    texts = [
        _text_with_outline(
            ax, x, y, _bold_math(label),
            ha="center", va="center", fontsize="small", fontweight="bold",
            color=color, zorder=10,
        )
        for label, x, y, color, _side in entries
    ]
    heights = [t.get_window_extent(renderer).height for t in texts]
    pad = 0.01 * axbb.height  # clearance kept between a label box and the line
    target_px = [
        ax.transData.transform((x, y))[1] + (h / 2 + pad) * (1 if side == "above" else -1)
        for (_label, x, y, _color, side), h in zip(entries, heights)
    ]
    placed_px = _spread_labels(target_px, heights, axbb.y0, axbb.y1)
    inv = ax.transData.inverted()
    for t, (_label, x, *_rest), py in zip(texts, entries, placed_px):
        t.set_position((x, inv.transform((axbb.x0, py))[1]))

    # Final pass: push any label still overlapping a curve, a marker or an earlier
    # label further out until its pixel box clears them.  The box only translates in
    # y, so candidates are tested by shifting its bounds rather than re-measuring text.
    obstacles = _curve_obstacles(ax)
    step = 2.0  # px
    clearance = 0.012 * axbb.height  # small gap kept between a label box and a line
    for t, (_label, _x, _y, _color, side) in zip(texts, entries):
        e = t.get_window_extent(renderer)
        half = e.height / 2
        base_cy = (e.y0 + e.y1) / 2
        lo_c, hi_c = axbb.y0 + half, axbb.y1 - half
        # Label box at vertical centre ``cy``, inflated by ``clearance`` so a label
        # stops a hair short of a curve, marker or neighbour rather than flush against
        # it.  Only y moves, so the x bounds stay fixed.
        def _clear_box(cy, e=e):
            return shapely.box(e.x0 - clearance, cy - half - clearance,
                               e.x1 + clearance, cy + half + clearance)

        if obstacles is not None and _clear_box(base_cy).intersects(obstacles):
            # Scan the preferred side first; if it is blocked to the axes edge (e.g. a
            # dot pinned against the floor) try the opposite side.
            sign = 1 if side == "above" else -1
            for direction in (sign, -sign):
                cy = base_cy + direction * step
                while lo_c <= cy <= hi_c and _clear_box(cy).intersects(obstacles):
                    cy += direction * step
                if lo_c <= cy <= hi_c:
                    t.set_position((t.get_position()[0], inv.transform((axbb.x0, cy))[1]))
                    break
        e = t.get_window_extent(renderer)
        box = shapely.box(e.x0, e.y0, e.x1, e.y1)
        obstacles = box if obstacles is None else shapely.union_all([obstacles, box])
    return texts
