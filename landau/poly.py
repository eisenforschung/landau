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

    @abc.abstractmethod
    def make(self, dd: pd.DataFrame, variables: list[str] = ["c", "T"]) -> Polygon:
        """Turn the subset of the full data belonging to one phase region into
        a polygon."""
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

    def make(self, dd, variables=["c", "T"]):
        if self.drop_interior and "border" in dd.columns:
            dd = dd.query("border")

        # concave hull algo seems more stable when both variables are of the same order
        pp = dd.sort_values(variables[0])[variables].to_numpy()
        pp = np.unique(pp[np.isfinite(pp).all(axis=-1)], axis=0)

        refnorm = {}
        for i, var in enumerate(variables):
            refnorm[var] = pp[:, i].min(), (np.ptp(pp[:, i]) or 1)
            pp[:, i] -= refnorm[var][0]
            pp[:, i] /= refnorm[var][1]
        points = shapely.MultiPoint(pp)
        # check for c-degenerate line phase
        shape = shapely.convex_hull(points)
        if variables[0] == "c" and isinstance(shape, shapely.LineString):
            coords = np.asarray(shape.coords)
            if np.allclose(coords[:, 0], coords[0, 0]):
                match refnorm["c"][0]:
                    case 0.0:
                        bias = +self.min_c_width / 2
                    case 1.0:
                        bias = -self.min_c_width / 2
                    case _:
                        bias = 0
                # artificially widen the line phase in c, so that we can make a
                # "normal" polygon for it.
                coords = np.concatenate(
                    [
                        # inverting the order for the second half of the array, makes
                        # it so that the points are in the correct order for the
                        # polygon
                        coords[::+1] - [self.min_c_width / 2, 0],
                        coords[::-1] + [self.min_c_width / 2, 0],
                    ],
                    axis=0,
                )
                coords[:, 0] += bias
        else:
            shape = shapely.concave_hull(points, ratio=self.ratio)
            if not isinstance(shape, shapely.Polygon):
                warn(f"Failed to construct polygon, got {shape} instead, skipping.")
                return None
            coords = np.asarray(shape.exterior.coords)
        for i, var in enumerate(variables):
            coords[:, i] *= refnorm[var][1]
            coords[:, i] += refnorm[var][0]
        return Polygon(coords)


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

    def make(self, td, variables=["c", "T"]):
        """
        Requires a grouped dataframe from get_transitions (by phase).
        """
        if "c" in variables and np.ptp(td.c) < self.min_c_width:
            meanc = td.c.mean()
            Tmin = td["T"].min()
            Tmax = td["T"].max()
            return Polygon(
                [
                    [meanc - self.min_c_width / 2, Tmin],
                    [meanc + self.min_c_width / 2, Tmin],
                    [meanc + self.min_c_width / 2, Tmax],
                    [meanc - self.min_c_width / 2, Tmax],
                ]
            )
        td = td.loc[ np.isfinite(td[variables[0]]) & np.isfinite(td[variables[1]]) ]
        sd = self._sort_segments(td, x_col=variables[0], y_col=variables[1])
        # sd = sd.loc[ np.isfinite(sd[variables[0]]) & np.isfinite(sd[variables[1]]) ]
        return Polygon(np.transpose([sd[v] for v in variables]))


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

        def make(self, dd, variables=["c", "T"]):
            c = dd.query('border')[variables].to_numpy()
            c = c[np.isfinite(c).all(axis=-1)]
            shape = shapely.convex_hull(shapely.MultiPoint(c))
            if isinstance(shape, shapely.LineString):
                coords = np.array(shape.buffer(self.min_c_width/2).exterior.coords)
                if "c" in variables:
                    match c[0, variables.index("c")]:
                        case 0.0:
                            bias = +self.min_c_width / 2
                        case 1.0:
                            bias = -self.min_c_width / 2
                        case _:
                            bias = 0
                coords[:, variables.index("c")] += bias
                return Polygon(coords)
            sc = StandardScaler().fit_transform(c)
            dm = pairwise_distances(sc)
            dm = (dm / dm[dm > 0].min()).round().astype(int)
            tour = solve_tsp_record_to_record(
                    dm, x0=np.argsort(np.arctan2(sc[:, 1], sc[:, 0])).tolist(),
                    max_iterations=self.max_iterations)[0]
            return Polygon(c[tour])
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

        def make(self, dd, variables=["c", "T"]):
            c = dd.query('border')[variables].to_numpy()
            c = c[np.isfinite(c).all(axis=-1)]
            shape = shapely.convex_hull(shapely.MultiPoint(c))
            if isinstance(shape, shapely.LineString):
                coords = np.array(shape.buffer(self.min_c_width/2).exterior.coords)
                if "c" in variables:
                    match c[0, variables.index("c")]:
                        case 0.0:
                            bias = +self.min_c_width / 2
                        case 1.0:
                            bias = -self.min_c_width / 2
                        case _:
                            bias = 0
                coords[:, variables.index("c")] += bias
                return Polygon(coords)
            sc = StandardScaler().fit_transform(c)
            dm = pairwise_distances(sc)
            dm = (dm / dm[dm > 0].min()).round().astype(int)
            tour = fast_tsp.find_tour(dm, self.duration_seconds)
            return Polygon(c[tour])

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
