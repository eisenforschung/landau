"""Methods to turn unstructured sets of points into polygons for plotting."""

import abc
from dataclasses import dataclass

import shapely
from python_tsp.heuristics import solve_tsp_record_to_record
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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
class PythonTsp(AbstractPolyMethod):
    """Find polygons by solving the Traveling Salesman Problem with the `python_tsp` module.

    Slower than the other methods but much more stable. Technically only solves an approximation to the TSP, but our
    phase boundaries should be well-behaved.
    """
    max_iterations: int = 10

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
        # alternative implementation in C++
        # seems more accurate than heuristics from python_tsp, but no conda package yet
        # import fast_tsp
        # tour = fast_tsp.find_tour(dm, .5)
        tour = solve_tsp_record_to_record(
                dm, x0=np.argsort(np.arctan2(sc[:, 1], sc[:, 0])).tolist(),
                max_iterations=self.max_iterations)[0]
        return Polygon(c[tour])


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
