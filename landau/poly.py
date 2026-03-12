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

from .calculate import get_transitions


@dataclass
class AbstractPolyMethod(abc.ABC):
    min_c_width: float = 0.01
    '''If line phases are detected, make them at least this thick in c space.'''

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Massage data set into format so that :method:`.make` can by applied
        over groups of columns `phase` and `phase_unit`."""
        return df

    def make(self, dd: pd.DataFrame, variables: list[str] = ["c", "T"]) -> Polygon | None:
        """Turn the subset of the full data belonging to one phase region into
        a polygon."""
        if getattr(self, "drop_interior", True) and "border" in dd.columns:
            dd = dd.query("border")

        pp = dd.sort_values(variables[0])[variables].to_numpy()
        pp = np.unique(pp[np.isfinite(pp).all(axis=-1)], axis=0)

        if len(pp) == 0:
            return None

        scaler = StandardScaler()
        pp_scaled = scaler.fit_transform(pp)

        # check for c-degenerate line phase
        points = shapely.MultiPoint(pp_scaled)
        shape = shapely.convex_hull(points)
        if not isinstance(shape, shapely.LineString):
            shape = self._make(pp_scaled)
            if shape is None:
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
        return Polygon(shape.exterior.coords)

    @abc.abstractmethod
    def _make(self, pp: np.ndarray) -> shapely.Geometry | None:
        """Turn the subset of the full data belonging to one phase region into
        a shapely geometry.  Expects a scaled array of coordinates."""
        pass

    def apply(self, df: pd.DataFrame, variables: list[str] = ["c", "T"]) -> pd.Series:
        return self.prepare(df).groupby(['phase', 'phase_unit']).apply(
                self.make, variables=variables

        ).dropna()


@dataclass
class Concave(AbstractPolyMethod):
    """Find polygons by constructing a concave hull around given points.

    Fast, but prone to unclean boundaries.
    """
    ratio: float = 0.1
    """Degree of "concave-ness", see `https://shapely.readthedocs.io/en/latest/reference/shapely.concave_hull.html <shapely>`_"""
    drop_interior: bool = True
    """Find concave set only of phase boundary points; usually helps to get the shape right, but can create holes."""

    def _make(self, pp: np.ndarray) -> shapely.Geometry | None:
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

    def make(self, dd: pd.DataFrame, variables: list[str] = ["c", "T"]) -> Polygon | None:
        """
        Requires a grouped dataframe from get_transitions (by phase).
        This method overrides the base `make` because it needs the unscaled DataFrame
        to properly sort segments using the `segment_label`.
        """
        if "c" in variables and np.ptp(dd.c) < self.min_c_width:
            meanc = dd.c.mean()
            Tmin = dd["T"].min()
            Tmax = dd["T"].max()
            shape = shapely.Polygon(
                [
                    [meanc - self.min_c_width / 2, Tmin],
                    [meanc + self.min_c_width / 2, Tmin],
                    [meanc + self.min_c_width / 2, Tmax],
                    [meanc - self.min_c_width / 2, Tmax],
                ]
            )
            return Polygon(shape.exterior.coords)
        td = dd.loc[ np.isfinite(dd[variables[0]]) & np.isfinite(dd[variables[1]]) ]
        if td.empty:
            return None
        sd = self._sort_segments(td, x_col=variables[0], y_col=variables[1])
        if sd.empty:
            return None
        coords = np.transpose([sd[v] for v in variables])
        if len(coords) < 3:
            return None
        shape = shapely.Polygon(coords)
        if shape.is_empty:
            return None
        shape = shape.buffer(self.min_c_width/2)
        return Polygon(shape.exterior.coords)

    def _make(self, pp: np.ndarray) -> shapely.Geometry | None:
        """Stub to satisfy AbstractPolyMethod."""
        raise NotImplementedError("Segments overrides make() and does not use _make().")


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

        def _make(self, pp: np.ndarray) -> shapely.Geometry | None:
            dm = pairwise_distances(pp)
            if not (dm > 0).any():
                return shapely.convex_hull(shapely.MultiPoint(pp))
            dm = (dm / dm[dm > 0].min()).round().astype(int)
            tour = solve_tsp_record_to_record(
                    dm, x0=np.argsort(np.arctan2(pp[:, 1], pp[:, 0])).tolist(),
                    max_iterations=self.max_iterations)[0]
            return shapely.Polygon(pp[tour]) if len(tour) > 2 else None
    __all__ += ["PythonTsp"]


with ImportAlarm("'fast-tsp' package required for FastTsp.  Install from pip.") as fast_tsp_alarm:
    import fast_tsp

    @dataclass
    class FastTsp(AbstractPolyMethod):
        """Find polygons by solving the Traveling Salesman Problem with the `fast_tsp` module.

        Much faster and higher quality than PythonTsp, but not yet on conda.
        """
        duration_seconds: float = 1.0
        """Maxixum time spent per search."""

        def _make(self, pp: np.ndarray) -> shapely.Geometry | None:
            dm = pairwise_distances(pp)
            if not (dm > 0).any():
                return shapely.convex_hull(shapely.MultiPoint(pp))
            dm = (dm / dm[dm > 0].min()).round().astype(int)
            tour = fast_tsp.find_tour(dm, self.duration_seconds)
            return shapely.Polygon(pp[tour]) if len(tour) > 2 else None

    __all__ += ["FastTsp"]


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
    if 'FastTsp' in __all__:
        allowed['fasttsp'] = FastTsp(**kwargs)
    if poly_method is None:
        if 'fasttsp' in allowed:
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
