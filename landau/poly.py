"""Methods to turn unstructured sets of points into polygons for plotting."""

import abc
from dataclasses import dataclass
from typing import ClassVar
from warnings import warn

from pyiron_snippets.import_alarm import ImportAlarm

import shapely
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from .calculate import get_transitions, _split_phase_unit


@dataclass
class AbstractPolyMethod(abc.ABC):
    min_c_width: float = 0.01
    '''If line phases are detected, make them at least this thick in c space.'''
    needs_unstable: ClassVar[bool] = False
    '''If True, :func:`~landau.plot.get_polygons` passes the whole ``keep_unstable=True``
    frame to :meth:`apply` instead of the clustered stable rows.'''

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Massage data set into format so that :method:`.make` can by applied
        over groups of columns `phase` and `phase_unit`."""
        return df

    def make(self, dd: pd.DataFrame, variables: list[str] = ["c", "T"]) -> shapely.Polygon | None:
        """Turn the subset of the full data belonging to one phase region into
        a buffered shapely polygon.  Conversion to matplotlib happens in
        :meth:`apply`."""
        border = dd["border"].to_numpy() if "border" in dd.columns else np.zeros(len(dd), dtype=bool)
        segment_label = dd["border_segment"].to_numpy() if "border_segment" in dd.columns else np.ones(len(dd), dtype=int)

        # To keep it simple, let's sort everything together
        idx = np.argsort(dd[variables[0]].to_numpy())
        pp = dd[variables].to_numpy()[idx]
        border = border[idx]
        segment_label = segment_label[idx]

        mask = np.isfinite(pp).all(axis=-1)
        pp = pp[mask]
        border = border[mask]
        segment_label = segment_label[mask]

        # np.unique is tricky with 3 parallel arrays. We can use np.unique on pp, returning indices
        _, unique_idx = np.unique(pp, axis=0, return_index=True)
        unique_idx.sort() # Keep original order somewhat
        pp = pp[unique_idx]
        border = border[unique_idx]
        segment_label = segment_label[unique_idx]

        if len(pp) == 0:
            return None

        scaler = StandardScaler()
        pp_scaled = scaler.fit_transform(pp)

        # check for c-degenerate line phase
        points = shapely.MultiPoint(pp_scaled)
        shape = shapely.convex_hull(points)
        if not isinstance(shape, shapely.LineString):
            shape = self._make(pp_scaled, border, segment_label)
            if shape is None:
                return None
            if isinstance(shape, shapely.Polygon) and not shape.is_valid:
                warn(f"{type(self).__name__}._make produced an invalid polygon "
                     f"({shapely.is_valid_reason(shape)}); repairing it.")
                shape = shapely.make_valid(shape, method="structure", keep_collapsed=False)
                if isinstance(shape, shapely.MultiPolygon):
                    shape = max(shape.geoms, key=shapely.area)
                elif not isinstance(shape, shapely.Polygon):
                    return None

        if shape.is_empty:
            return None

        if isinstance(shape, shapely.Polygon):
            coords = np.asarray(shape.exterior.coords)
            if len(coords) < 3:
                return None
            shape = shapely.Polygon(scaler.inverse_transform(coords))
        elif isinstance(shape, shapely.LineString):
            shape = shapely.LineString(scaler.inverse_transform(np.asarray(shape.coords)))
        else:
            return None

        shape = shape.buffer(self.min_c_width/2)
        if isinstance(shape, shapely.MultiPolygon):
            shape = max(shape.geoms, key=shapely.area)
            warn("polymethod returned disjoined polygons, returning largest.")
        return shape

    @abc.abstractmethod
    def _make(self, pp: np.ndarray, border: np.ndarray, segment_label: np.ndarray) -> shapely.Geometry | None:
        """Turn the subset of the full data belonging to one phase region into
        a shapely geometry.  Expects a scaled array of coordinates."""
        pass

    def apply(self, df: pd.DataFrame, variables: list[str] = ["c", "T"]) -> pd.Series:
        shapes = self.prepare(df).groupby(['phase', 'phase_unit']).apply(
                self.make, variables=variables, include_groups=False

        ).dropna()
        shapes = shapes[~shapes.map(lambda s: s.is_empty)]
        trimmed = self._trim_overlaps(shapes)
        return trimmed.map(self._to_mpl_polygon).dropna()

    def _trim_overlaps(self, shapes: pd.Series) -> pd.Series:
        """Symmetrically subtract pairwise overlap between buffered polygons.

        Each polygon was inflated by ``min_c_width/2`` so small-solubility
        phases stay visible.  Where adjacent buffered phases touch this
        creates an overlap strip; here we subtract from every polygon its
        neighbours' un-buffered shapes so the seam lands on the original
        shared boundary.
        """
        if len(shapes) < 2:
            return shapes
        r = self.min_c_width / 2
        out: dict = {}
        for k, a in shapes.items():
            trimmed = a
            for k2, b in shapes.items():
                if k2 == k:
                    continue
                if not trimmed.intersects(b):
                    continue
                b_orig = b.buffer(-r)
                if b_orig.is_empty:
                    continue
                try:
                    new = trimmed.difference(b_orig)
                except shapely.errors.GEOSException:
                    continue
                if not new.is_empty:
                    trimmed = new
                if isinstance(trimmed, shapely.MultiPolygon):
                    trimmed = self._collapse_parts(trimmed)
            out[k] = trimmed
        return pd.Series(out, name=shapes.name).reindex(shapes.index)

    def _collapse_parts(self, shape: shapely.MultiPolygon) -> shapely.Geometry:
        """What to keep of a region that trimming split apart: its largest part."""
        return max(shape.geoms, key=shapely.area)

    @staticmethod
    def _to_mpl_polygon(shape: shapely.Geometry) -> Polygon | None:
        if not isinstance(shape, shapely.Polygon) or shape.is_empty:
            return None
        coords = np.asarray(shape.exterior.coords)
        if len(coords) < 3:
            return None
        return Polygon(coords)


@dataclass
class Concave(AbstractPolyMethod):
    """Find polygons by constructing a concave hull around given points.

    Fast, but prone to unclean boundaries.
    """
    ratio: float = 0.1
    """Degree of "concave-ness", see `https://shapely.readthedocs.io/en/latest/reference/shapely.concave_hull.html <shapely>`_"""
    drop_interior: bool = True
    """Find concave set only of phase boundary points; usually helps to get the shape right, but can create holes."""

    def _make(self, pp: np.ndarray, border: np.ndarray, segment_label: np.ndarray) -> shapely.Geometry | None:
        if self.drop_interior:
            pp = pp[border]
        if len(pp) == 0:
            return None
        points = shapely.MultiPoint(pp)
        try:
            shape = shapely.concave_hull(points, ratio=self.ratio)
        except shapely.errors.GEOSException:
            # Degenerate point clouds (collinear/coincident/denormal coordinates)
            # can make GEOS fail to locate a vertex; such a set has no polygon.
            warn("concave_hull failed on a degenerate point set, skipping.")
            return None
        if not isinstance(shape, shapely.Polygon):
            warn(f"Failed to construct polygon, got {shape} instead, skipping.")
            return None
        return shape


@dataclass
class Segments(AbstractPolyMethod):
    """Construct polygons by identifying phase boundaries and stitching them together in a poor man's TSP approach.

    Requires that phase diagram data was generated with `refine=True`.

    FIXME: sort_segment should just set up a distance matrix for the segments and use python_tsp on those."""

    def prepare(self, df):
        if "refined" not in df.columns:
            raise ValueError("Segments methods requires refined phase boundaries!")
        df.loc[:, "phase"] = df.phase_id
        tdf = get_transitions(df)
        tdf["phase"], tdf["phase_unit"] = _split_phase_unit(tdf["phase"])
        return tdf

    @staticmethod
    def _sort_segments(df, x_col="c", y_col="T", segment_label="border_segment"):
        """
        Sorts the points in df such that they can be used as the bounding polygon of a phase in a binary diagram.

        Assumptions:
        1. df contains only data on a single, coherent phase, i.e. the c/T points are "connected"

        Algorithm:
        1. Subset the data according to the column given by `segment_label`.  These should label connected points on a single two-phase boundary. Such a subset is called a segment.
        2. Sort points in each segment by a 1D PCA. (Sorting by c or T alone fails when the segment is either vertical or horizontal.)
        3. Sort the segments so that they "easily" fit together:
            a. Pick the segment with minimum `x` as the "head"
            b. Go over all other segments, s, and:
                b0. Get the distance from endpoint of "head" to either the starting point or the end point of s
                b1. if the distance to the end point is shorter than to the starting point, invert order of s
                b2. return the minimum of both distances
            c. the segment with smallest distance to the current "head" is the next "head" and removed from the pool of segments
            d. break if no segments left
        4. return the segments in the order they were picked as "head"s.

        a) is a heuristic for "normal" phase diagrams, starting from the left (or right) we can often make a full circle.
        Picking a random segments breaks for phases that are stable at the lower or upper edge of the diagram, where we technically do not compute
        a "segment".  A "proper" fix would be to modify b to allow joining also to the start of "head" rather than just the end.
        """
        if df.empty:
            return pd.DataFrame(columns=df.columns)

        norm = np.ptp(df[[x_col, y_col]], axis=0).values
        norm = np.where(norm == 0, 1, norm)

        # Step 1: PCA Projection
        def pca_projection(group):
            # avoid warnings when clustering only found one or two points
            if len(group) < 2:
                return group
            pca = PCA(n_components=1)
            projected = pca.fit_transform(group[[x_col, y_col]])
            group["projected"] = projected
            return group.sort_values("projected").copy().drop("projected", axis="columns").reset_index(drop=True)

        segments = []
        for label, dd in df.groupby(segment_label):
            segments.append(pca_projection(dd))

        if not segments:
            return pd.DataFrame(columns=df.columns)

        ordered = _greedy_stitch(segments, norm, x_col, y_col)
        return pd.concat(ordered, ignore_index=True)

    def _make(self, pp: np.ndarray, border: np.ndarray, segment_label: np.ndarray) -> shapely.Geometry | None:
        """
        Requires a grouped dataframe from get_transitions (by phase).
        """
        if np.all(segment_label == 1):
            raise ValueError("Segments methods requires refined phase boundaries (segment_label must be provided)!")

        # Segments currently assumes variable names "c" and "T" for _sort_segments
        # but the actual values can just be named "x" and "y" for sorting.
        # Create a temporary DataFrame to reuse _sort_segments
        td = pd.DataFrame({
            "x": pp[:, 0],
            "y": pp[:, 1] if pp.shape[1] > 1 else np.zeros(len(pp)),
            "border_segment": segment_label
        })

        sd = self._sort_segments(td, x_col="x", y_col="y", segment_label="border_segment")
        if sd.empty:
            return None

        coords = sd[["x", "y"]].to_numpy()
        if len(coords) < 3:
            return None

        return shapely.Polygon(coords)


def _greedy_stitch(
        segments: list[pd.DataFrame],
        norm: np.ndarray,
        x_col: str,
        y_col: str,
) -> list[pd.DataFrame]:
    """Greedy nearest-neighbour ordering of pre-sorted border segments.

    Starts from the segment with the smallest ``x_col`` value, then repeatedly
    picks from the remaining segments the one whose closer endpoint is nearest
    to the current head's tail. The picked segment is reversed in place when
    its end is closer than its start. Distances are scaled by ``norm`` so that
    ``x_col`` and ``y_col`` contribute on comparable scales.

    Returns the segments in stitch order; each segment is the original
    DataFrame, possibly with row order reversed. The caller concatenates.

    Notes
    -----
    The min-``x_col`` head heuristic is documented to fail when a phase is
    stable at the upper or lower edge of the diagram (where no proper
    "segment" is computed); see :class:`Segments`.
    """
    if not segments:
        return []

    def endpoint(s, where):
        return s.iloc[where][[x_col, y_col]]

    def scaled_dist(p1, p2):
        return np.linalg.norm((p2 - p1) / norm)

    def flip(s):
        s.reset_index(drop=True, inplace=True)
        s.loc[:] = s.loc[::-1].reset_index(drop=True)
        return s

    def find_distance(head, segment):
        head_tail = endpoint(head, -1)
        head2tail = scaled_dist(head_tail, endpoint(segment, 0))
        tail2tail = scaled_dist(head_tail, endpoint(segment, -1))
        if tail2tail < head2tail:
            flip(segment)
            return tail2tail
        return head2tail

    head, *remaining = sorted(segments, key=lambda s: s[x_col].min())
    ordered = [head]
    while remaining:
        head, *remaining = sorted(remaining, key=lambda s: find_distance(head, s))
        ordered.append(head)
    return ordered


def _pca_sort_segment(pts: np.ndarray) -> np.ndarray:
    """Sort the points of a single border segment along its principal axis."""
    if len(pts) < 2:
        return pts
    pca = PCA(n_components=1)
    proj = pca.fit_transform(pts).ravel()
    return pts[np.argsort(proj)]


def _segments_from_labels(pp: np.ndarray, segment_label: np.ndarray) -> list[np.ndarray]:
    """Group `pp` by `segment_label` and PCA-sort each group."""
    segments = []
    for lab in np.unique(segment_label):
        pts = pp[segment_label == lab]
        if len(pts) == 0:
            continue
        segments.append(_pca_sort_segment(pts))
    return segments


def _segment_tsp_polygon(
        segments: list[np.ndarray],
        solve_tour,
) -> shapely.Geometry | None:
    """Stitch already-sorted border segments into a polygon by solving a TSP on
    segment endpoints.

    Each segment contributes two nodes (its two endpoints) to a 2N-node TSP.
    The intra-segment edge is given distance 0 so the optimum tour walks each
    segment end-to-end; inter-segment distances are the actual Euclidean
    distance between endpoints.  Hence the per-pair segment-to-segment cost
    incurred by the tour is the minimum over the four endpoint pairings, which
    is the desired notion of "segment distance".

    `solve_tour` is a callable taking an integer distance matrix and returning a
    list of node indices forming the tour.
    """
    if len(segments) == 0:
        return None
    if len(segments) == 1:
        coords = segments[0]
        if len(coords) < 3:
            return None
        return shapely.Polygon(coords)

    n = len(segments)
    endpoints = np.array([[s[0], s[-1]] for s in segments]).reshape(2 * n, -1)
    dm = pairwise_distances(endpoints)
    for i in range(n):
        dm[2 * i, 2 * i + 1] = 0
        dm[2 * i + 1, 2 * i] = 0

    pos = dm[dm > 0]
    if len(pos) == 0:
        return shapely.convex_hull(shapely.MultiPoint(np.vstack(segments)))
    dm_int = (dm / pos.min()).round().astype(int)

    tour = list(solve_tour(dm_int))

    # The tour is cyclic, so the solver is free to return a rotation whose
    # ends fall inside a segment (the zero-cost intra-segment edge being the
    # wrap-around). The first-encounter rule below would then walk that
    # segment in the wrong direction relative to its neighbours and produce a
    # self-intersecting ring, so rotate off the intra-segment edge first.
    while tour[0] // 2 == tour[-1] // 2:
        tour.append(tour.pop(0))

    seen = set()
    chunks = []
    for node in tour:
        seg = node // 2
        if seg in seen:
            continue
        seen.add(seg)
        # node is even -> entered segment at its start, walk forward
        # node is odd  -> entered segment at its end, walk backward
        chunks.append(segments[seg] if node % 2 == 0 else segments[seg][::-1])

    coords = np.vstack(chunks)
    if len(coords) < 3:
        return None
    return shapely.Polygon(coords)


def _split_rings(verts, codes):
    """Closed rings of a matplotlib path given as (vertices, codes)."""
    rings, start = [], None
    for i, code in enumerate(codes):
        if code == 1:  # MOVETO
            start = i
        elif code == 79:  # CLOSEPOLY
            rings.append(verts[start:i + 1])
    return [r for r in rings if len(r) >= 4]


def _contour_rings(pts: np.ndarray, z: np.ndarray, level: float):
    """Rings bounding ``z <= level`` on the Delaunay triangulation of ``pts``, and the triangulation."""
    from matplotlib.tri import Triangulation
    from matplotlib._tri import TriContourGenerator

    tri = Triangulation(pts[:, 0], pts[:, 1])
    gen = TriContourGenerator(tri.get_cpp_triangulation(), z)
    verts, codes = gen.create_filled_contour(-1.0, level)
    return [r for v, c in zip(verts, codes) for r in _split_rings(v, c)], tri


@dataclass
class Contour(AbstractPolyMethod):
    """Find polygons by contouring each phase's ``dphi`` in (mu, T).

    Needs a ``calc_phase_diagram(..., keep_unstable=True)`` frame.  A phase's
    ``dphi`` is zero where it is stable and positive elsewhere, so the rings
    bounding ``dphi = 0`` on the triangulation of the phase's own (mu, T) rows
    enclose its stable regions; stable phase regions tile the (mu, T) plane, so
    the rings come out ordered.  Each ring is then replaced by the phase's own
    border rows (and rows on the edge of the sampled range) in the ring's order,
    which puts the vertices on the refined boundary and gives them the phase's
    own coordinates in ``variables``, e.g. ``c`` for a c-T diagram.  Disconnected
    regions of one phase come out as separate rings, so no clustering is needed.

    The synthetic ``mu = +-inf`` terminal rows are placed one sampling step
    beyond the grid in mu, so regions reach the pure components.  Refined rows
    traced beyond the sampled range are dropped: there is no grid around them to
    contour against.

    A miscibility gap puts one phase on both sides of the gap line.  Its two rows
    are moved off the line by ``gap_shift`` (fraction of the mu range), each
    towards its own side, and the line itself, interpolated ``gap_densify`` times
    between gap rows, is marked as not stable, so the gap becomes a channel the
    contour runs around.
    """

    gap_shift: float = 3e-3
    """Offset of miscibility-gap rows from the gap line, as a fraction of the sampled mu range."""
    gap_densify: int = 8
    """Points per gap-row interval marking the gap line; keeps triangles from bridging the channel."""

    needs_unstable: ClassVar[bool] = True

    def _make(self, pp, border, segment_label):
        raise NotImplementedError("Contour builds all regions of a phase at once in apply().")

    def apply(self, df: pd.DataFrame, variables: list[str] = ["c", "T"]) -> pd.Series:
        if "dphi" not in df.columns:
            raise ValueError("Contour needs a calc_phase_diagram(..., keep_unstable=True) frame (dphi column).")
        grid = df[(df["locus"] == "interior") & np.isfinite(df["mu"])]
        lo = grid[["mu", "T"]].min().to_numpy(float)
        span = grid[["mu", "T"]].max().to_numpy(float) - lo
        span[span == 0] = 1.0
        mus = np.unique(grid["mu"])
        step = np.diff(mus).min() if len(mus) > 1 else span[0]

        df = df.copy()
        df.loc[df["mu"] == -np.inf, "mu"] = lo[0] - step
        df.loc[df["mu"] == np.inf, "mu"] = lo[0] + span[0] + step
        xy = (df[["mu", "T"]].to_numpy(float) - lo) / span
        pad = step / span[0] + 1e-9
        inside = (np.isfinite(xy).all(axis=1)
                  & (xy[:, 0] >= -pad) & (xy[:, 0] <= 1 + pad)
                  & (xy[:, 1] >= -1e-9) & (xy[:, 1] <= 1 + 1e-9))
        df = df[inside]
        # stable rows are on or inside their region: zeroes the refiners' tolerance on
        # refined rows and gives the terminal rows (dphi NaN) their value
        df.loc[df["stable"], "dphi"] = 0.0
        # Signed distance to the region: a phase's dphi minus the lowest dphi of the other
        # rows at the same (mu, T).  Negative inside, positive outside, zero on refined
        # boundary rows, so its zero crossing interpolates between competing phases rather
        # than hugging the stable rows.  Rows alone at their (mu, T) (terminal rows) get 0.
        key = [df["mu"], df["T"]]
        first = df.groupby(key)["dphi"].transform("min")
        runner_up = df["dphi"].where(df["dphi"] > first).groupby(key).transform("min")
        tied = (df["dphi"] == first).groupby(key).transform("sum") > 1
        second = first.where(tied, runner_up)
        df["dphi"] = np.where(df["dphi"] > first, df["dphi"] - first, first - second)
        df["dphi"] = df["dphi"].fillna(0.0)

        shapes = {}
        for phase, dd in df.groupby("phase"):
            for unit, shape in enumerate(self._phase_regions(dd, lo, span, variables)):
                shapes[(phase, unit)] = shape
        if not shapes:
            return pd.Series(dtype=object)
        shapes = pd.Series(shapes).rename_axis(["phase", "phase_unit"])
        shapes = shapes[~shapes.map(lambda s: s.is_empty)]
        trimmed = self._trim_overlaps(shapes)
        # a region trimming split apart keeps all its parts, each as its own unit
        parts = {}
        for (phase, _), shape in trimmed.items():
            for g in getattr(shape, "geoms", [shape]):
                if not g.is_empty:
                    parts[(phase, sum(k[0] == phase for k in parts))] = g
        parts = pd.Series(parts, dtype=object).rename_axis(["phase", "phase_unit"])
        return parts.map(self._to_mpl_polygon).dropna()

    def _collapse_parts(self, shape):
        return shape

    def _phase_regions(self, dd, lo, span, variables):
        dd = dd[np.isfinite(dd["dphi"])].copy()
        gap = dd.duplicated(["mu", "T"], keep=False)
        marks = pd.DataFrame({"mu": [], "T": [], "dphi": []})
        if gap.any():
            line = dd.loc[gap, ["mu", "T"]].drop_duplicates().sort_values("T")
            k = np.linspace(0, len(line) - 1, max(len(line) - 1, 0) * self.gap_densify + 1)
            idx = np.arange(len(line))
            marks = pd.DataFrame({"mu": np.interp(k, idx, line["mu"]), "T": np.interp(k, idx, line["T"]), "dphi": 1.0})
            side = np.sign(dd.loc[gap, "c"] - dd[gap].groupby(["mu", "T"])["c"].transform("mean"))
            dd.loc[gap, "mu"] = dd.loc[gap, "mu"] + side * self.gap_shift * span[0]
            dd.loc[gap, "border"] = True
        pts = pd.concat([dd[["mu", "T", "dphi"] + [v for v in variables if v not in ("mu", "T")]], marks],
                        ignore_index=True)
        xy = (pts[["mu", "T"]].to_numpy(float) - lo) / span
        rings, tri = _contour_rings(xy, pts["dphi"].to_numpy(float), level=1e-12)

        own = dd[dd["stable"]]
        oxy = (own[["mu", "T"]].to_numpy(float) - lo) / span
        on_edge = (oxy.min(axis=1) < 1e-9) | (oxy.max(axis=1) > 1 - 1e-9)
        cand = own.loc[own["border"].to_numpy(bool) | on_edge, variables]
        cand_xy = oxy[own["border"].to_numpy(bool) | on_edge]
        # Where a boundary leaves the sampled range (or the terminal column) between two rows,
        # the contour crosses the range's edge there; that crossing is a vertex too, its
        # variables interpolated on the triangulation.
        walls = shapely.union(shapely.MultiPoint(xy).convex_hull.exterior, shapely.box(0, 0, 1, 1).exterior)
        ring_xy = np.concatenate(rings) if rings else np.empty((0, 2))
        ring_pts = shapely.points(ring_xy)
        cross = ring_xy[(shapely.distance(walls, ring_pts) < 1e-9)
                        & (shapely.distance(shapely.MultiPoint(cand_xy), ring_pts) > 1e-9)]
        if len(cross):
            from matplotlib.tri import LinearTriInterpolator
            vals = {}
            for v in variables:
                if v in ("mu", "T"):
                    i = ("mu", "T").index(v)
                    vals[v] = cross[:, i] * span[i] + lo[i]
                else:
                    interp = LinearTriInterpolator(tri, pts[v].to_numpy(float))
                    vals[v] = np.ma.filled(interp(cross[:, 0], cross[:, 1]), np.nan)
            extra = pd.DataFrame(vals)
            ok = np.isfinite(extra.to_numpy(float)).all(axis=1)
            cand = pd.concat([cand, extra[ok]], ignore_index=True)
            cand_xy = np.concatenate([cand_xy, cross[ok]])
        cand_pts = shapely.points(cand_xy)

        # filled-contour rings are exteriors and holes; a hole is a ring inside an exterior
        rings = sorted((shapely.Polygon(r) for r in rings), key=lambda q: -q.area)
        parent = []
        for i, q in enumerate(rings):
            inner = q.representative_point()
            parent.append(next((j for j in range(i) if parent[j] is None and rings[j].contains(inner)), None))
        mapped = [self._along_ring(q.exterior, cand, cand_pts, variables) for q in rings]
        out = []
        for i, (shape, p) in enumerate(zip(mapped, parent)):
            if p is not None or shape is None:
                continue
            holes = [mapped[j] for j, pj in enumerate(parent) if pj == i and mapped[j] is not None]
            if holes:
                shape = shape.difference(shapely.union_all([h.buffer(0) for h in holes]))
            shape = shape.buffer(self.min_c_width / 2)
            # regions that touch only along a collapsed edge come apart here; each is its own region
            out.extend(g for g in getattr(shape, "geoms", [shape]) if not g.is_empty)
        return out

    @staticmethod
    def _along_ring(ring, cand, cand_pts, variables):
        """The candidate rows on ``ring``, in ring order, as a geometry in ``variables``."""
        on = shapely.distance(ring, cand_pts) < 1e-7
        if on.sum() < 2:
            return None
        order = np.argsort(shapely.line_locate_point(ring, cand_pts[on]), kind="stable")
        coords = cand[on].to_numpy(float)[order]
        shape = shapely.Polygon(coords) if len(coords) >= 3 else shapely.LineString(coords)
        if isinstance(shape, shapely.Polygon) and shape.area > 0:
            return shapely.make_valid(shape) if not shape.is_valid else shape
        # a line phase: all vertices share one c, so the ring collapses onto a line
        return shapely.LineString(coords)


__all__ = ["Concave", "Segments", "Contour"]


with ImportAlarm("'python_tsp' package required for PythonTsp.  Install from conda or pip.") as python_tsp_alarm:
    from python_tsp.heuristics import solve_tsp_record_to_record

    @dataclass
    class PythonTsp(AbstractPolyMethod):
        """Find polygons by solving the Traveling Salesman Problem with the `python_tsp` module.

        Slower than the other methods but much more stable. Technically only solves an approximation to the TSP, but our
        phase boundaries should be well-behaved.
        """
        max_iterations: int = 10
        retries: int = 2
        """How often to re-solve with a 10x larger iteration budget when the tour self-intersects.

        A self-intersecting tour survives only repaired, which can cut off parts of the phase region,
        so spending more solver effort first is worth it."""

        def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
            if df.shape[0] > 50_000:
                warn("Large number of sample points! PythonTsp may be very slow, "
                     "try FastTsp or one of the other polygon methods.")
            return df

        def _make(self, pp: np.ndarray, border: np.ndarray, segment_label: np.ndarray) -> shapely.Geometry | None:
            pp = pp[border]
            if len(pp) == 0:
                return None
            dm = pairwise_distances(pp)
            if not (dm > 0).any():
                return shapely.convex_hull(shapely.MultiPoint(pp))
            dm = (dm / dm[dm > 0].min()).round().astype(int)
            x0 = np.argsort(np.arctan2(pp[:, 1], pp[:, 0])).tolist()
            iterations = self.max_iterations
            for _ in range(self.retries + 1):
                tour = solve_tsp_record_to_record(dm, x0=x0, max_iterations=iterations)[0]
                if len(tour) <= 2:
                    return None
                shape = shapely.Polygon(pp[tour])
                if shape.is_valid:
                    break
                iterations *= 10
            return shape

    @dataclass
    class SegmentPythonTsp(Segments):
        """Like :class:`Segments`, but stitch the (PCA-sorted) border segments
        together by solving a TSP on segment endpoints with `python_tsp`.

        The distance between two segments is the minimum of the four
        endpoint-to-endpoint distances; this is achieved by a 2N-node TSP
        formulation with zero-cost intra-segment edges.
        """
        max_iterations: int = 10

        def _make(self, pp, border, segment_label):
            if np.all(segment_label == 1):
                raise ValueError("SegmentPythonTsp requires refined phase boundaries (segment_label must be provided)!")
            segments = _segments_from_labels(pp, segment_label)

            def solve(dm_int):
                return solve_tsp_record_to_record(dm_int, max_iterations=self.max_iterations)[0]

            return _segment_tsp_polygon(segments, solve)

    __all__ += ["PythonTsp", "SegmentPythonTsp"]


with ImportAlarm("'fast-tsp' package required for FastTsp.  Install from pip.") as fast_tsp_alarm:
    import fast_tsp

    @dataclass
    class FastTsp(AbstractPolyMethod):
        """Find polygons by solving the Traveling Salesman Problem with the `fast_tsp` module.

        Much faster and higher quality than PythonTsp, but not yet on conda.
        """
        duration_seconds: float = 1.0
        """Maxixum time spent per search."""

        def _make(self, pp: np.ndarray, border: np.ndarray, segment_label: np.ndarray) -> shapely.Geometry | None:
            pp = pp[border]
            if len(pp) == 0:
                return None
            dm = pairwise_distances(pp)
            if not (dm > 0).any():
                return shapely.convex_hull(shapely.MultiPoint(pp))
            dm = (dm / dm[dm > 0].min()).round().astype(int)
            tour = fast_tsp.find_tour(dm, self.duration_seconds)
            return shapely.Polygon(pp[tour]) if len(tour) > 2 else None

    @dataclass
    class SegmentFastTsp(Segments):
        """Like :class:`Segments`, but stitch the (PCA-sorted) border segments
        together by solving a TSP on segment endpoints with `fast_tsp`.

        The distance between two segments is the minimum of the four
        endpoint-to-endpoint distances; this is achieved by a 2N-node TSP
        formulation with zero-cost intra-segment edges.
        """
        duration_seconds: float = 0.01

        def _make(self, pp, border, segment_label):
            if np.all(segment_label == 1):
                raise ValueError("SegmentFastTsp requires refined phase boundaries (segment_label must be provided)!")
            segments = _segments_from_labels(pp, segment_label)

            def solve(dm_int):
                return fast_tsp.find_tour(dm_int, self.duration_seconds)

            return _segment_tsp_polygon(segments, solve)

    __all__ += ["FastTsp", "SegmentFastTsp"]


@fast_tsp_alarm
@python_tsp_alarm
def handle_poly_method(poly_method, **kwargs):
    '''Uniform handling of poly_method between plot_phase_diagram and plot_mu_phase_diagram.
    Some **kwargs trickery required to handle now deprecated min_c_width and alpha arguments.'''
    ratio = kwargs.pop('ratio', kwargs.pop('alpha', Concave.ratio))
    allowed = {
                'concave': Concave(**kwargs, ratio=ratio),
                'segments': Segments(**kwargs),
                'contour': Contour(**kwargs),
    }
    if 'PythonTsp' in __all__:
        allowed['tsp'] = PythonTsp(**kwargs)
        allowed['segment-tsp'] = SegmentPythonTsp(**kwargs)
    if 'FastTsp' in __all__:
        allowed['fasttsp'] = FastTsp(**kwargs)
        allowed['segment-fasttsp'] = SegmentFastTsp(**kwargs)
    if poly_method is None:
        if 'segment-fasttsp' in allowed:
            poly_method = 'segment-fasttsp'
        elif 'segment-tsp' in allowed:
            poly_method = 'segment-tsp'
        elif 'fasttsp' in allowed:
            poly_method = 'fasttsp'
        elif 'tsp' in allowed:
            poly_method = 'tsp'
        else:
            poly_method = 'concave'
    if isinstance(poly_method, str):
        try:
            return allowed[poly_method]
        except KeyError:
            raise ValueError(f"poly_method must be one of: {list(allowed.keys())}!") from None
    if not isinstance(poly_method, AbstractPolyMethod):
        raise TypeError("poly_method must be recognized str or AbstractPolyMethod!")
    return poly_method


__all__ += ["handle_poly_method"]
