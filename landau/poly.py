"""Methods to turn unstructured sets of points into polygons for plotting."""

import abc
from dataclasses import dataclass

import pandas as pd
from matplotlib.patches import Polygon


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
