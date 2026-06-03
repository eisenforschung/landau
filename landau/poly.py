"""Methods to turn unstructured sets of points into polygons for plotting."""

import abc
from dataclasses import dataclass
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
                shape = shapely.make_valid(shape)
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
                    trimmed = max(trimmed.geoms, key=shapely.area)
            out[k] = trimmed
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
    ratio = kwargs.pop('ratio', kwargs.pop('alpha', Concave.ratio))
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
