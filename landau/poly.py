"""Methods to turn unstructured sets of points into polygons for plotting."""

import abc
from dataclasses import dataclass
from warnings import warn

from pyiron_snippets.import_alarm import ImportAlarm

import shapely
import shapely.affinity
import shapely.ops
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from .calculate import get_transitions


@dataclass
class AbstractPolyMethod(abc.ABC):
    min_c_width: float = 0.01
    '''If line phases are detected, make them at least this thick in c space.'''

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
                     f"({shapely.is_valid_reason(shape)}); plotting will buffer it but "
                     f"the result may be inaccurate.")

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
        """Symmetrically resolve overlap between buffered phase polygons.

        Each shape was inflated by ``min_c_width/2`` so small-solubility
        phases stay visible.  Where neighbours touch this creates an
        overlap strip; the visible seam should fall at the locus
        equidistant from the *un-buffered* originals (their generalised
        Voronoi bisector).

        For every overlapping pair (a, b) we walk each connected
        component of their buffered union, sample its exterior boundary
        uniformly, label each sample by the closer un-buffered original,
        binary-search each label transition for the exact seam endpoint,
        route the cut polyline through any shared geometry of the two
        originals (a tangent corner, a coincident edge -- the apex of a
        eutectic is at distance 0 from both, so the seam must pass
        through it), then split the component and hand each piece to its
        closer original.

        Falls back to a plain difference where the bisector cut cannot
        be constructed (one phase fully covers the component, the split
        is geometrically degenerate, or a phase is too thin for erosion).
        """
        if len(shapes) < 2:
            return shapes
        r = self.min_c_width / 2
        out = dict(shapes.items())
        originals = {k: s.buffer(-r) for k, s in out.items()}

        keys = list(out.keys())
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                a, b = out[ki], out[kj]
                if not a.intersects(b):
                    continue
                inter = a.intersection(b)
                if inter.is_empty or inter.area == 0:
                    continue
                oa, ob = originals[ki], originals[kj]
                if oa.is_empty or ob.is_empty:
                    if not oa.is_empty:
                        out[kj] = b.difference(oa)
                    if not ob.is_empty:
                        out[ki] = a.difference(ob)
                    continue
                out[ki], out[kj] = _voronoi_split(a, b, oa, ob, inter, r)

        for k, v in out.items():
            if isinstance(v, shapely.MultiPolygon):
                out[k] = max(v.geoms, key=shapely.area)
        return pd.Series(out, name=shapes.name).reindex(shapes.index)

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
        shape = shapely.concave_hull(points, ratio=self.ratio)
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
        tdf["phase_unit"] = tdf.phase.str.rsplit('_', n=1).map(lambda x: int(x[1]))
        tdf["phase"] = tdf.phase.str.rsplit('_', n=1).map(lambda x: x[0])
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

        com = df[[x_col, y_col]].mean()
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

        # initial sorting by center of mass angle
        segments = sorted(
                segments,
                key=lambda s: np.arctan2( (s[y_col].mean() - com[y_col]) / norm[1],
                                          (s[x_col].mean() - com[x_col]) / norm[0])
        )

        def start(s):
            return s.iloc[0][[x_col, y_col]]

        def end(s):
            return s.iloc[-1][[x_col, y_col]]

        def dist(p1, p2):
            return np.linalg.norm((p2 - p1) / norm)

        def flip(s):
            s.reset_index(drop=True, inplace=True)
            s.loc[:] = s.loc[::-1].reset_index(drop=True)
            return s

        head, *remaining = sorted(segments, key=lambda s: s[x_col].min())

        def find_distance(head, segment):
            head2tail = dist(end(head), start(segment))
            tail2tail = dist(end(head), end(segment))
            if tail2tail < head2tail:
                flip(segment)
                return tail2tail
            else:
                return head2tail

        segments = [head]
        while len(remaining) > 0:
            head, *remaining = sorted(remaining, key=lambda s: find_distance(head, s))
            segments.append(head)

        return pd.concat(segments, ignore_index=True)

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


def _nudge_outward(
        p: tuple[float, float], neighbour: tuple[float, float], step: float,
) -> tuple[float, float]:
    """Push ``p`` by ``step`` along the unit vector from ``neighbour`` to ``p``.

    Used to extend the bisector-cut endpoints just past the polygon
    boundary so :func:`shapely.ops.split` actually divides the polygon
    rather than silently returning it intact.
    """
    dx, dy = p[0] - neighbour[0], p[1] - neighbour[1]
    n = (dx * dx + dy * dy) ** 0.5
    if n == 0:
        return p
    return (p[0] + dx / n * step, p[1] + dy / n * step)


def _ordered_shared_coords(geom: shapely.Geometry) -> list[tuple[float, float]]:
    """Flatten an intersection geometry to ``(x, y)`` tuples in geometric
    order along the shared feature (point, linestring, ...)."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(geom.x, geom.y)]
    if geom.geom_type == "MultiPoint":
        return [(g.x, g.y) for g in geom.geoms]
    if geom.geom_type == "LineString":
        return list(geom.coords)
    if geom.geom_type == "MultiLineString":
        return [c for g in geom.geoms for c in g.coords]
    if geom.geom_type == "GeometryCollection":
        return [c for g in geom.geoms for c in _ordered_shared_coords(g)]
    return []


def _shared_anchors(
        oa: shapely.Polygon, ob: shapely.Polygon, tol: float,
) -> list[tuple[float, float]]:
    """Pin points the bisector cut should pass through.

    Uses ``oa.intersection(ob)`` directly when the two originals truly
    share geometry (a coincident edge or vertex).  If that intersection
    is empty but the two are within ``tol`` of each other -- as happens
    at point-tangent apices when ``oa`` and ``ob`` were recovered via
    ``buffered.buffer(-r)`` and the corner rounded slightly off -- falls
    back to the midpoint of the closest-approach pair.
    """
    coords = _ordered_shared_coords(oa.intersection(ob))
    if coords:
        return coords
    p, q = shapely.ops.nearest_points(oa, ob)
    if p.distance(q) < tol:
        return [(0.5 * (p.x + q.x), 0.5 * (p.y + q.y))]
    return []


def _bisector_cut(
        comp: shapely.Polygon, oa: shapely.Polygon, ob: shapely.Polygon,
        shared_pts: list[tuple[float, float]], n_samples: int = 200,
) -> shapely.LineString | None:
    """Polyline that splits ``comp`` along the bisector of ``oa`` / ``ob``.

    Samples ``comp`` exterior, labels each point by the closer original,
    binary-searches each label transition to pin it exactly where
    ``d(oa) == d(ob)`` on the boundary, then routes the cut through
    ``shared_pts`` so the seam reaches the equidistance locus interior
    to ``comp``.  Returns ``None`` when no usable two-endpoint seam is
    found (one phase dominates the whole component, or there are more
    than two transitions which we don't try to resolve here).
    """
    ext = comp.exterior
    L = ext.length
    ts = np.linspace(0, L, n_samples, endpoint=False)

    # scale-normalise so x and y contribute comparably to the distance
    minx, miny, maxx, maxy = comp.bounds
    sx = 1.0 / max(maxx - minx, 1e-12)
    sy = 1.0 / max(maxy - miny, 1e-12)
    oa_s = shapely.affinity.scale(oa, xfact=sx, yfact=sy, origin=(0, 0))
    ob_s = shapely.affinity.scale(ob, xfact=sx, yfact=sy, origin=(0, 0))
    pts = np.array([(p.x, p.y) for p in (ext.interpolate(t) for t in ts)])
    pts_s = shapely.points(pts[:, 0] * sx, pts[:, 1] * sy)
    label = (shapely.distance(ob_s, pts_s) < shapely.distance(oa_s, pts_s)).astype(int)

    trans_idx = np.where(label != np.roll(label, -1))[0]
    if len(trans_idx) != 2:
        return None

    seam_pts = []
    for t in trans_idx:
        lo, hi = ts[t], ts[(t + 1) % len(ts)]
        if hi < lo:
            hi += L
        lbl_lo = label[t]
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            p = ext.interpolate(mid % L)
            ps = shapely.Point(p.x * sx, p.y * sy)
            lbl_mid = int(ob_s.distance(ps) < oa_s.distance(ps))
            if lbl_mid == lbl_lo:
                lo = mid
            else:
                hi = mid
        p = ext.interpolate(0.5 * (lo + hi) % L)
        seam_pts.append((p.x, p.y))

    if shared_pts:
        s0 = shapely.Point(seam_pts[0])
        ordered = sorted(shared_pts, key=lambda c: s0.distance(shapely.Point(c)))
        path = [seam_pts[0], *ordered, seam_pts[1]]
    else:
        path = list(seam_pts)

    overshoot = max(maxx - minx, maxy - miny) * 1e-6
    path[0] = _nudge_outward(path[0], path[1], overshoot)
    path[-1] = _nudge_outward(path[-1], path[-2], overshoot)
    return shapely.LineString(path)


def _voronoi_split(
        a: shapely.Polygon, b: shapely.Polygon,
        oa: shapely.Polygon, ob: shapely.Polygon,
        inter: shapely.Geometry, r: float,
) -> tuple[shapely.Polygon, shapely.Polygon]:
    """Trim ``(a, b)`` so each keeps only its half of every overlap.

    For each connected component of the buffered union, build a bisector
    cut polyline (:func:`_bisector_cut`) and remove the foreign-side
    piece from the wrong polygon.  Falls back to subtracting the other's
    un-buffered original when the cut can't be constructed.  ``r`` is
    the buffer radius used to derive ``oa, ob``; it sets the tolerance
    for treating closest-approach as a true tangent point.
    """
    a_out, b_out = a, b
    shared = _shared_anchors(oa, ob, tol=r)
    union = shapely.union(a, b)
    comps = list(union.geoms) if hasattr(union, "geoms") else [union]
    for comp in comps:
        if not comp.intersects(inter):
            continue
        cut = _bisector_cut(comp, oa, ob, shared)
        if cut is None:
            a_out = a_out.difference(ob)
            b_out = b_out.difference(oa)
            continue
        try:
            pieces = shapely.ops.split(comp, cut).geoms
        except Exception:
            a_out = a_out.difference(ob)
            b_out = b_out.difference(oa)
            continue
        for piece in pieces:
            if piece.is_empty or not piece.is_valid or piece.area == 0:
                continue
            rp = piece.representative_point()
            if oa.distance(rp) < ob.distance(rp):
                b_out = b_out.difference(piece)
            else:
                a_out = a_out.difference(piece)
    return a_out, b_out


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

    tour = solve_tour(dm_int)

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


__all__ = ["Concave", "Segments"]


with ImportAlarm("'python_tsp' package required for PythonTsp.  Install from conda or pip.") as python_tsp_alarm:
    from python_tsp.heuristics import solve_tsp_record_to_record

    @dataclass
    class PythonTsp(AbstractPolyMethod):
        """Find polygons by solving the Traveling Salesman Problem with the `python_tsp` module.

        Slower than the other methods but much more stable. Technically only solves an approximation to the TSP, but our
        phase boundaries should be well-behaved.
        """
        max_iterations: int = 10

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
            tour = solve_tsp_record_to_record(
                    dm, x0=np.argsort(np.arctan2(pp[:, 1], pp[:, 0])).tolist(),
                    max_iterations=self.max_iterations)[0]
            return shapely.Polygon(pp[tour]) if len(tour) > 2 else None

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
    ratio = kwargs.pop('alpha', Concave.ratio)
    allowed = {
                'concave': Concave(**kwargs, ratio=ratio),
                'segments': Segments(**kwargs),
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
