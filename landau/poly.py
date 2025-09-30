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
