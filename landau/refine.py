"""
Strategies for refining a coarsely-sampled phase diagram.

A :class:`Refiner` is a single, self-contained algorithm that proposes candidate
regions of the (T, mu) plane that likely bracket a phase transition, solves
each one to locate the exact transition, and emits dataframe rows tagged with
``refined=<label>``.

The orchestrator :func:`refine_phase_diagram` (in :mod:`landau.calculate`) picks
a default list of refiners based on whether the input samples vary along mu,
T or both, but users can pass any sequence of :class:`Refiner` instances to
swap or extend the default behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterator, Mapping, Sequence

import warnings

import numpy as np
import pandas as pd
import scipy.optimize as so
from scipy.spatial import Delaunay

from .phases import Phase


__all__ = [
    "RefinedPoint",
    "Refiner",
    "ScanRefiner",
    "DelaunayLineRefiner",
    "DelaunayTripleRefiner",
    "default_refiners",
]


@dataclass(frozen=True)
class RefinedPoint:
    """A located transition point at which ``phases`` coexist."""

    T: float
    mu: float
    phases: tuple[str, ...]


def _state_row(phase: Phase, T: float, mu: float) -> dict:
    return {
        "T": T,
        "mu": mu,
        "phi": phase.semigrand_potential(T, mu),
        "c": phase.concentration(T, mu),
        "phase": phase.name,
    }


def _expand_rows(pt: RefinedPoint, phases: Mapping[str, Phase]) -> list[dict]:
    return [_state_row(phases[name], pt.T, pt.mu) for name in pt.phases]


def _dominated(pt: RefinedPoint, phases: Mapping[str, Phase]) -> bool:
    """True if any phase outside ``pt.phases`` has lower potential at ``pt``."""
    phi = phases[pt.phases[0]].semigrand_potential(pt.T, pt.mu)
    return any(
        p.semigrand_potential(pt.T, pt.mu) < phi
        for p in phases.values()
        if p.name not in pt.phases
    )


def _find_one_point(phase1, phase2, potential, var_range):
    """Root-find where the two phases have equal potential along a 1D parameter."""
    return so.root_scalar(
        lambda x: potential(phase1, x) - potential(phase2, x),
        bracket=var_range,
        x0=np.mean(var_range),
        xtol=1e-6,
    ).root


class Refiner(ABC):
    """Strategy interface for one refinement pass."""

    label: ClassVar[str]

    @abstractmethod
    def propose(self, df: pd.DataFrame):
        """Yield candidate objects to be handed to :meth:`solve`."""

    @abstractmethod
    def solve(self, cand, phases: Mapping[str, Phase]) -> RefinedPoint | None:
        """Locate an exact transition for one candidate; return None to skip."""

    def run(self, df: pd.DataFrame, phases: Mapping[str, Phase]) -> pd.DataFrame:
        rows: list[dict] = []
        for cand in self.propose(df):
            pt = self.solve(cand, phases)
            if pt is None:
                continue
            if pt.T < 0:
                continue
            if _dominated(pt, phases):
                continue
            rows.extend(_expand_rows(pt, phases))
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["stable"] = True
        out["border"] = True
        out["refined"] = self.label
        return out


# -- 1D scan refiners ---------------------------------------------------------


@dataclass(frozen=True)
class _ScanCandidate:
    phase1: str
    phase2: str
    T: float
    mu: float
    bracket: tuple[float, float]


class ScanRefiner(Refiner):
    """
    Walk samples sorted along one axis (``mu`` or ``T``) and root-find the
    exact transition between every pair of adjacent samples that have
    different stable phases.

    Operates on each unique value of the orthogonal axis independently.
    """

    def __init__(self, by: str):
        assert by in ("mu", "T"), "by must be 'mu' or 'T'"
        self.by = by
        self.other = "T" if by == "mu" else "mu"
        self.label = by

    def propose(self, df: pd.DataFrame) -> Iterator[_ScanCandidate]:
        for other_val, group in df.groupby(self.other):
            g = group.sort_values(self.by).reset_index(drop=True)
            change = g.index[g.phase != g.phase.shift(-1).ffill()]
            for i in change:
                r1, r2 = g.loc[i], g.loc[i + 1]
                bracket = tuple(sorted([r1[self.by], r2[self.by]]))
                yield _ScanCandidate(
                    phase1=r1.phase, phase2=r2.phase,
                    T=r1["T"], mu=r1["mu"], bracket=bracket,
                )

    def solve(self, cand: _ScanCandidate, phases) -> RefinedPoint | None:
        p1, p2 = phases[cand.phase1], phases[cand.phase2]
        if self.by == "mu":
            mu = _find_one_point(
                p1, p2,
                lambda p, x: p.semigrand_potential(cand.T, x),
                cand.bracket,
            )
            return RefinedPoint(T=cand.T, mu=mu, phases=(cand.phase1, cand.phase2))
        else:
            T = _find_one_point(
                p1, p2,
                lambda p, x: p.semigrand_potential(x, cand.mu),
                cand.bracket,
            )
            return RefinedPoint(T=T, mu=cand.mu, phases=(cand.phase1, cand.phase2))


# -- Delaunay-based refiners --------------------------------------------------


def _delaunay_simplices(df: pd.DataFrame):
    """Yield (simplex_df, phase_count) for each simplex of the (mu, T) tess."""
    dela = Delaunay(df[["mu", "T"]])
    phases_arr = df.phase.to_numpy()[dela.simplices]
    counts = np.array([len(set(x)) for x in phases_arr])
    for simplex, n in zip(dela.simplices, counts):
        yield df.iloc[simplex], n


@dataclass(frozen=True)
class _SimplexCandidate:
    simplex: pd.DataFrame  # 3 rows of the input df


class DelaunayLineRefiner(Refiner):
    """
    For every Delaunay simplex containing exactly two phases, locate the
    transition along the line from the lone vertex (the "peak") to the
    midpoint of the two same-phase vertices (the "base").

    This makes the (technically unjustified) assumption that the transition
    crosses that line; for a more accurate scheme along the actual triangle
    edges, write a new refiner.
    """

    label = "delaunay"

    def propose(self, df: pd.DataFrame) -> Iterator[_SimplexCandidate]:
        for simplex, n in _delaunay_simplices(df):
            if n == 2:
                yield _SimplexCandidate(simplex=simplex)

    def solve(self, cand: _SimplexCandidate, phases) -> RefinedPoint | None:
        cand_df = cand.simplex
        p1_xy, p2_xy = cand_df.groupby("phase")[["T", "mu"]].mean().to_numpy()
        name1, name2 = cand_df.phase.unique()
        phase1, phase2 = phases[name1], phases[name2]

        def project(t):
            T, mu = p1_xy + (p2_xy - p1_xy) * t
            return T, mu

        try:
            t = _find_one_point(
                phase1, phase2,
                lambda phase, x: phase.semigrand_potential(*project(x)),
                (0, 1),
            )
        except ValueError:
            warnings.warn(
                f"Failed to refine triangle between {p1_xy} and {p2_xy} "
                f"of phases {cand_df.phase.unique()}!",
                stacklevel=2,
            )
            return None
        T, mu = project(t)
        return RefinedPoint(T=T, mu=mu, phases=(name1, name2))


class DelaunayTripleRefiner(Refiner):
    """
    For every Delaunay simplex containing three distinct phases, locate the
    triple point by minimizing the sum of pairwise potential differences,
    starting from the simplex centroid.
    """

    label = "delaunay-triple"

    def propose(self, df: pd.DataFrame) -> Iterator[_SimplexCandidate]:
        for simplex, n in _delaunay_simplices(df):
            if n == 3:
                yield _SimplexCandidate(simplex=simplex)

    def solve(self, cand: _SimplexCandidate, phases) -> RefinedPoint | None:
        tr = cand.simplex
        T0, mu0 = tr[["T", "mu"]].mean()
        names = tuple(tr.phase.unique())
        p1, p2, p3 = (phases[n] for n in names)

        def triplemin(x):
            T, mu = x
            phi1 = p1.semigrand_potential(T, mu)
            phi2 = p2.semigrand_potential(T, mu)
            phi3 = p3.semigrand_potential(T, mu)
            return abs(phi1 - phi2) + abs(phi2 - phi3) + abs(phi3 - phi1)

        T, mu = so.fmin(triplemin, (T0, mu0), disp=False)
        return RefinedPoint(T=T, mu=mu, phases=names)


# -- Defaults -----------------------------------------------------------------


def default_refiners(df: pd.DataFrame) -> Sequence[Refiner]:
    """Pick the standard refiners based on which axes are sampled."""
    multiple_mus = len(df["mu"].unique()) > 1
    multiple_ts = len(df["T"].unique()) > 1
    if multiple_mus and multiple_ts:
        return [DelaunayLineRefiner(), DelaunayTripleRefiner()]
    refiners: list[Refiner] = []
    if multiple_mus:
        refiners.append(ScanRefiner(by="mu"))
    if multiple_ts:
        refiners.append(ScanRefiner(by="T"))
    return refiners
