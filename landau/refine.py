"""
Strategies for refining a coarsely-sampled phase diagram.

A :class:`Refiner` is a single, self-contained algorithm that proposes
candidate regions of the ``(T, mu)`` plane that likely bracket a phase
transition, locates the exact transition inside each one, and emits
dataframe rows tagged with ``refined=<label>``.

The orchestrator :func:`landau.calculate.refine_phase_diagram` picks a
default list of refiners with :func:`default_refiners` based on which
axes the input samples (``mu``, ``T``, or both), but you can pass any
sequence of :class:`Refiner` instances to swap or extend it.

Refiners shipped here
---------------------

* :class:`ScanRefiner` — 1-D bisection between adjacent samples
  whose stable phase label disagrees. Used when only one of ``mu`` or
  ``T`` is sampled.
* :class:`DelaunayLineRefiner` — one transition per Delaunay simplex
  that spans two phases.
* :class:`DelaunayTripleRefiner` — locate triple points from three-
  phase simplices.
* :class:`ClausiusClapeyronRefiner` — *predictor-corrector* trace of
  a two-phase coexistence line; seeds from each two-phase simplex
  and walks the line in both T directions.
* :class:`MiscibilityGapRefiner` — same predictor-corrector idea but
  for an intra-phase miscibility gap (a single phase that splits
  into two coexisting compositions). Seeds from single-phase
  simplices with a wide c-spread.

The two Clausius-Clapeyron refiners share their skeleton via the
private :class:`_CCBase` ABC; see the comment block right above it
for a quick algorithm overview aimed at new readers.

Writing a new refiner
---------------------

Subclass :class:`Refiner` and implement ``propose`` and ``solve``.
``propose`` yields candidate dataclasses; ``solve`` turns one
candidate into zero or more :class:`RefinedPoint` /
:class:`RefinedMiscibilityGap`. The base class's ``run`` then
filters out negative-T or dominated points and packages the rest
into the output dataframe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import ClassVar, Iterable, Iterator, Mapping, Sequence

import warnings

import numpy as np
import pandas as pd
import scipy.optimize as so
import shapely
from scipy.spatial import Delaunay

from .features import Locus
from .phases import Phase


# -- Type aliases ------------------------------------------------------------
#
# Used across the module to keep nested-tuple annotations readable.
#
#   Bracket      = (lo, hi) endpoints of a 1-D range.
#   TMuPoint     = a single (T, mu) location.
#   Trace        = one completed coexistence trace, T-sorted.
#   TraceList    = all completed traces for a single coexistence line
#                  (per phase pair / per miscibility-gap phase).

Bracket = tuple[float, float]
TMuPoint = tuple[float, float]
Trace = tuple[TMuPoint, ...]
TraceList = tuple[Trace, ...]


__all__ = [
    "RefinedPoint",
    "RefinedMiscibilityGap",
    "Refiner",
    "ScanRefiner",
    "DelaunayLineRefiner",
    "DelaunayTripleRefiner",
    "ClausiusClapeyronRefiner",
    "MiscibilityGapRefiner",
    "default_refiners",
]


@dataclass(frozen=True)
class RefinedPoint:
    """A located transition point at which ``phases`` coexist.

    Attributes
    ----------
    T : float
        Temperature of the transition (K).
    mu : float
        Coexistence chemical potential (eV).
    phases : tuple[str, ...]
        Names of the phases that coexist here. Length 2 for a regular
        two-phase boundary, length 3 for a triple point.
    boundary_id : int
        Identifier shared by all rows that belong to the same coexistence
        line (assigned by the refiner's ``run()``).

    :meth:`to_rows` tags each emitted row with ``locus``:
    :attr:`~landau.features.Locus.TRIPLE` for three coexisting phases,
    :attr:`~landau.features.Locus.BOUNDARY` otherwise.
    """

    T: float
    mu: float
    phases: tuple[str, ...]
    boundary_id: int = 0

    def phase_names(self) -> set[str]:
        return set(self.phases)

    def to_rows(self, phases: Mapping[str, Phase]) -> list[dict]:
        rows = [_state_row(phases[name], self.T, self.mu) for name in self.phases]
        locus = Locus.TRIPLE if len(self.phases) == 3 else Locus.BOUNDARY
        for row in rows:
            row["boundary_id"] = self.boundary_id
            row["locus"] = locus
        return rows


@dataclass(frozen=True)
class RefinedMiscibilityGap:
    """A coexistence point inside a single phase (miscibility gap).

    The single ``phase`` splits into two compositions at coexistence
    chemical potential ``mu``. :meth:`to_rows` emits two rows that
    share that exact ``mu`` value but carry the pre-computed
    branch concentrations ``c_left`` and ``c_right`` directly — so
    the two branches show up as distinct ``(c, T)`` points while
    still sitting on the exact coexistence chemical potential. Both
    rows are tagged ``locus = Locus.BOUNDARY``.

    Attributes
    ----------
    T : float
        Temperature of the coexistence (K).
    mu : float
        Coexistence chemical potential (eV). Both emitted rows carry
        this exact value.
    phase : str
        Name of the splitting phase.
    c_left, c_right : float
        Concentrations of the two coexisting branches, taken straight
        from the locating scan. Stored on the dataclass rather than
        re-queried via ``ph.concentration(T, mu)`` because for some
        phases (e.g. brute-grid minimisers) the re-query can collapse
        to a single branch even though the scan resolved both.
    """

    T: float
    mu: float
    phase: str
    c_left: float
    c_right: float
    boundary_id: int = 0

    def phase_names(self) -> set[str]:
        return {self.phase}

    def to_rows(self, phases: Mapping[str, Phase]) -> list[dict]:
        ph = phases[self.phase]
        phi = float(ph.semigrand_potential(self.T, self.mu))
        return [
            {"T": self.T, "mu": self.mu, "phi": phi,
             "c": self.c_left,  "phase": ph.name, "boundary_id": self.boundary_id,
             "locus": Locus.BOUNDARY},
            {"T": self.T, "mu": self.mu, "phi": phi,
             "c": self.c_right, "phase": ph.name, "boundary_id": self.boundary_id,
             "locus": Locus.BOUNDARY},
        ]


def _state_row(phase: Phase, T: float, mu: float) -> dict:
    return {
        "T": T,
        "mu": mu,
        "phi": phase.semigrand_potential(T, mu),
        "c": phase.concentration(T, mu),
        "phase": phase.name,
    }


def _dominated(pt, phases: Mapping[str, Phase]) -> bool:
    """True iff some phase outside ``pt.phase_names()`` has a lower
    semigrand potential at ``(pt.T, pt.mu)`` — meaning the refined
    transition we found isn't actually globally stable, so we drop it.
    """
    own = pt.phase_names()
    own_phase = next(iter(own))
    own_phi = phases[own_phase].semigrand_potential(pt.T, pt.mu)
    return any(
        p.semigrand_potential(pt.T, pt.mu) < own_phi
        for p in phases.values()
        if p.name not in own
    )


def _find_one_point(phase1, phase2, potential, var_range):
    """Find the 1-D root of ``potential(phase1, x) - potential(phase2, x)``.

    Thin wrapper around :func:`scipy.optimize.root_scalar` used by every
    refiner that needs to locate where two phases share a potential
    along a single parameter (mu at fixed T, T at fixed mu, or a
    convex combination along a simplex edge). The caller passes a
    bracket ``var_range`` over which the difference is known to change
    sign.
    """
    return so.root_scalar(
        lambda x: potential(phase1, x) - potential(phase2, x),
        bracket=var_range,
        x0=np.mean(var_range),
        xtol=1e-6,
    ).root


class Refiner(ABC):
    """Strategy interface for one refinement pass.

    A subclass implements two halves of the pipeline:

    * ``propose(df)`` walks the coarse samples and yields *candidate*
      objects — neutral data carriers describing a region of
      ``(T, mu)`` space that likely contains an exact transition.
    * ``solve(cand, phases)`` turns one candidate into zero or more
      :class:`RefinedPoint` / :class:`RefinedMiscibilityGap`.

    :meth:`run` glues them together: it filters out negative-T or
    dominated points, expands each refined transition to its rows, and
    packages the result as a dataframe tagged with ``refined=label``.
    Subclasses set ``label`` as a class attribute.
    """

    label: ClassVar[str]

    @abstractmethod
    def propose(self, df: pd.DataFrame):
        """Yield candidate objects to be handed to :meth:`solve`."""

    @abstractmethod
    def solve(
        self, cand, phases: Mapping[str, Phase]
    ) -> Iterable[RefinedPoint]:
        """Locate exact transitions for one candidate.

        Return an iterable (typically a list) of :class:`RefinedPoint`
        or :class:`RefinedMiscibilityGap`. Returning an empty iterable
        is fine and just means "this candidate produced nothing".
        """

    def run(self, df: pd.DataFrame, phases: Mapping[str, Phase]) -> pd.DataFrame:
        """propose → solve → filter → dataframe."""
        rows: list[dict] = []
        boundary_id = 0
        for cand in self.propose(df):
            pts = [pt for pt in self.solve(cand, phases)
                   if pt.T >= 0 and not _dominated(pt, phases)]
            for pt in pts:
                rows.extend(replace(pt, boundary_id=boundary_id).to_rows(phases))
            if pts:
                boundary_id += 1
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
    bracket: Bracket


class ScanRefiner(Refiner):
    """
    Walk samples sorted along one axis (``mu`` or ``T``) and root-find the
    exact transition between every pair of adjacent samples that have
    different stable phases.

    Operates on each unique value of the orthogonal axis independently.

    Parameters
    ----------
    by : {"mu", "T"}
        Axis to walk along. The orthogonal axis is grouped, and within
        each group adjacent samples whose stable phase changes seed a
        bisection along ``by``.
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

    def solve(self, cand: _ScanCandidate, phases) -> list[RefinedPoint]:
        if self.by == "mu":
            def potential(p, x):
                return p.semigrand_potential(cand.T, x)

            def point(x, pair):
                return RefinedPoint(T=cand.T, mu=x, phases=pair)
        else:
            def potential(p, x):
                return p.semigrand_potential(x, cand.mu)

            def point(x, pair):
                return RefinedPoint(T=x, mu=cand.mu, phases=pair)
        return self._solve_pair(cand.phase1, cand.phase2, cand.bracket, potential, point, phases)

    def _solve_pair(self, name1, name2, bracket, potential, point, phases) -> list[RefinedPoint]:
        """Root-find the (name1, name2) crossing inside ``bracket``.

        When a third phase is more stable at the crossing, the pair does not
        actually coexist there: the stable phase changes name1 → dominator →
        name2 inside the bracket, with the dominator's stable window narrower
        than the sampling grid.  Recurse into the two sub-brackets so both real
        transitions are found instead of leaving the candidate to be dropped
        by :func:`_dominated` in :meth:`Refiner.run`.
        """
        x = _find_one_point(phases[name1], phases[name2], potential, bracket)
        rivals = {p.name: potential(p, x) for p in phases.values() if p.name not in (name1, name2)}
        if rivals:
            dom = min(rivals, key=rivals.get)
            if rivals[dom] < potential(phases[name1], x):
                return (
                    self._solve_pair(name1, dom, (bracket[0], x), potential, point, phases)
                    + self._solve_pair(dom, name2, (x, bracket[1]), potential, point, phases)
                )
        return [point(x, (name1, name2))]


# -- Delaunay-based refiners --------------------------------------------------


def _delaunay_simplices(df: pd.DataFrame):
    """Yield (simplex_df, phase_count) for each simplex of the (mu, T) tess."""
    dela = Delaunay(df[["mu", "T"]])
    phases_arr = df.phase.to_numpy()[dela.simplices]
    counts = np.array([len(set(x)) for x in phases_arr])
    for simplex, n in zip(dela.simplices, counts):
        yield df.iloc[simplex], n


def _phase_centroids_xy(simplex: pd.DataFrame) -> tuple[TMuPoint, TMuPoint]:
    """``((T, mu), (T, mu))`` of each phase's vertex centroids.

    For a 2-phase simplex (3 vertices, one phase appears once and the
    other twice) the line from one centroid to the other is guaranteed
    to cross the phase boundary inside the simplex, which gives
    :class:`ClausiusClapeyronRefiner` a reliable seed bracket.
    """
    by_phase = simplex.groupby("phase")[["T", "mu"]].mean().to_numpy()
    return tuple(by_phase[0]), tuple(by_phase[1])


def _simplex_containment(point: TMuPoint, simplex: pd.DataFrame) -> float:
    """Smallest barycentric coordinate of ``(T, mu)`` w.r.t. the triangle.

    ``>= 0`` iff the point lies inside (or on an edge of) the simplex; the
    more negative the value the further outside it is. Barycentric
    coordinates are affine-invariant, so the value does not depend on the
    anisotropy between the T and mu axes, and it doubles as a "how nearly
    contained" score: the simplex with the largest value owns the point.
    Returns ``-inf`` for a degenerate (zero-area) simplex so it never wins.
    """
    (x1, y1), (x2, y2), (x3, y3) = simplex[["T", "mu"]].to_numpy()
    x, y = point
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if det == 0:
        return float("-inf")
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    c = 1.0 - a - b
    return min(a, b, c)


@dataclass(frozen=True)
class _SimplexCandidate:
    simplex: pd.DataFrame  # 3 rows of the input df


@dataclass(frozen=True)
class _TripleCandidate:
    """One three-phase simplex plus all of its siblings in the tessellation.

    Carrying the sibling simplices lets :meth:`DelaunayTripleRefiner.solve`
    decide, as a pure function of the candidate, which simplex owns the
    located triple point without any cross-call dedup state.
    """

    simplex: pd.DataFrame  # 3 rows of the input df
    siblings: tuple  # every three-phase simplex, including ``simplex``


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

    def solve(self, cand: _SimplexCandidate, phases) -> list[RefinedPoint]:
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
            return []
        T, mu = project(t)
        return [RefinedPoint(T=T, mu=mu, phases=(name1, name2))]


class DelaunayTripleRefiner(Refiner):
    """
    For every Delaunay simplex containing three distinct phases, locate the
    triple point by minimizing the sum of pairwise potential differences,
    starting from the simplex centroid.

    The located minimum may fall outside every three-phase simplex---inside a
    neighbouring two-phase triangle (the Delaunay partition has at most one
    triangle that strictly contains any point). Ownership is therefore
    determined by :func:`_simplex_containment`: the sibling with the largest
    score (least far outside the minimum) is the owner. :meth:`solve` emits
    only when its own simplex is that owner, so exactly one candidate fires per
    triple point. The siblings travel in the candidate, keeping :meth:`solve` a
    pure function of ``(cand, phases)`` — no ``__init__``, no dedup state, no
    ``run`` override.
    """

    label = "delaunay-triple"

    def propose(self, df: pd.DataFrame) -> Iterator[_TripleCandidate]:
        siblings = tuple(simplex for simplex, n in _delaunay_simplices(df) if n == 3)
        for simplex in siblings:
            yield _TripleCandidate(simplex=simplex, siblings=siblings)

    def solve(self, cand: _TripleCandidate, phases) -> list[RefinedPoint]:
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
        owner = max(cand.siblings, key=lambda s: _simplex_containment((T, mu), s))
        if owner is not tr:
            return []
        return [RefinedPoint(T=T, mu=mu, phases=names)]


# -- Clausius-Clapeyron tracers ----------------------------------------------
#
# Both Clausius-Clapeyron refiners (ClausiusClapeyronRefiner and
# MiscibilityGapRefiner) trace a coexistence line through (T, mu)
# space with the same predictor-corrector skeleton, factored into
# `_CCBase` below. The skeleton, in pseudocode:
#
#   for each candidate from propose(df):
#       if simplex straddles an already-traced segment of this line:
#           skip                                                  (dedup)
#       (T0, mu0) = _seed_step(cand)         # land on the line   (seed)
#       emit at (T0, mu0)
#       walk T upward and downward from T0:
#           predict mu_next = _predict_mu(mu_star, dmu/dT, dT)    (predict)
#           refine mu_next via _refine_step at T_next              (correct)
#           update dmu/dT, emit at (T_next, mu_next)
#           dT = clip(_dT_adapt(...), dT_min, dT_max)              (adapt)
#       stop on T-bound, ValueError from _refine_step (e.g. gap
#       closed), or when the walker steps onto a prior trace.
#
# Subclasses only implement propose / _seed_step / _refine_step /
# _emit / _pair_key, plus optional _predict_mu and _dT_adapt overrides.


# -- Shared geometry / bookkeeping helpers -----------------------------------


def _simplex_brackets(simplex: pd.DataFrame):
    """Bounding box and centroid T of a Delaunay simplex.

    Returns ``(mu_lo, mu_hi, T_lo, T_hi, T_seed)``; both refiners use
    these as the seed simplex's mu/T extents and the seed temperature.
    """
    return (
        float(simplex["mu"].min()),
        float(simplex["mu"].max()),
        float(simplex["T"].min()),
        float(simplex["T"].max()),
        float(simplex["T"].mean()),
    )


def _trace_geom(points: list[TMuPoint]):
    """One trace -> one shapely geometry. ``Point`` for length 1,
    ``LineString`` otherwise."""
    if len(points) == 1:
        return shapely.Point(*points[0])
    return shapely.LineString(points)


def _point_on_line(T: float, mu: float,
                   traces: TraceList, tol_mu: float) -> bool:
    """Is ``(T, mu)`` within ``tol_mu`` of any traced segment?

    ``traces`` is one entry per previously completed trace on the same
    coexistence line; each entry is a tuple of ``(T, mu)`` points.
    Each trace becomes its own shapely geometry (Point or LineString)
    and we test the query point against each independently — concatenating
    them into a single line would invent spurious cross-segments
    between disparate traces.

    Parameters
    ----------
    T, mu : float
        Query point.
    traces : tuple of tuples
        Outer dimension = traces; inner dimension = ``(T, mu)`` points of
        one trace. May be empty.
    tol_mu : float
        Euclidean-distance tolerance. For a typical near-horizontal
        coexistence line this is essentially the vertical mu offset.
    """
    if not traces:
        return False
    p = shapely.Point(T, mu)
    return any(p.distance(_trace_geom(list(tr))) <= tol_mu for tr in traces)


def _simplex_straddles(cand, traces: TraceList) -> bool:
    """True if any previously traced segment passes within one bracket
    width of the candidate simplex's (T, mu) bounding box.

    The bbox is inflated by the simplex's own width on each axis so
    that a converged mu* sitting one scan-grid spacing outside a
    neighbouring simplex (the common case when two simplices share a
    grid edge at the coexistence line) is still detected.

    ``traces`` is a list-of-traces (one entry per completed solve()
    call). Each trace is tested independently; concatenating them
    would invent spurious cross-segments that block unrelated
    simplices at the same temperature.
    """
    if not traces:
        return False
    mu_lo, mu_hi = cand.mu_bracket
    T_lo, T_hi = cand.T_bracket
    mu_pad = mu_hi - mu_lo
    T_pad = T_hi - T_lo
    bbox = shapely.box(T_lo - T_pad, mu_lo - mu_pad,
                       T_hi + T_pad, mu_hi + mu_pad)
    return any(bbox.intersects(_trace_geom(list(tr))) for tr in traces)


# -- Predictor-corrector tracer base class -----------------------------------


@dataclass(frozen=True)
class _StepResult:
    """One isothermal refinement step.

    ``mu_star`` is the converged coexistence chemical potential.
    ``extra`` is a refiner-specific payload (a value, dataclass, or
    ``None``) read back by :meth:`_CCBase._emit` and
    :meth:`_CCBase._dT_adapt`; the base class itself never inspects it.
    """

    mu_star: float
    extra: object = None


class _CCBase(Refiner):
    """Shared predictor-corrector trace skeleton.

    Both Clausius-Clapeyron refiners walk T in two directions from a
    seed point, predicting mu* by linear extrapolation along
    ``dmu*/dT`` and correcting via an isothermal refinement inside a
    bracket centred on the prediction. This base implements that walk
    plus the per-phase-pair straddle dedup; subclasses only have to
    supply ``propose`` plus four small hooks
    (``_seed_step``, ``_refine_step``, ``_emit``, ``_pair_key``) and
    optionally override ``_predict_mu`` / ``_dT_adapt`` for a fancier
    predictor or step scaling.

    A subclass's ``_refine_step`` may raise :class:`ValueError` to
    abort the trace cleanly (used by the miscibility-gap refiner to
    stop when the gap closes).

    Parameters
    ----------
    dT_max : float
        Maximum allowed temperature step (K).
    dT_min : float
        Minimum allowed temperature step (K) and the bootstrap step
        right after the seed.
    max_steps : int
        Hard cap on steps per trace direction, just in case the
        adaptive logic somehow fails to terminate.
    """

    def __init__(self, dT_max: float = 50.0, dT_min: float = 1.0,
                 max_steps: int = 500):
        self.dT_max = dT_max
        self.dT_min = dT_min
        self.max_steps = max_steps

    # -- subclass hooks -----------------------------------------------------

    @abstractmethod
    def _seed_step(self, cand, phases) -> tuple[float, _StepResult] | None:
        """Find a starting point on the coexistence line.

        Parameters
        ----------
        cand
            One propose() candidate, of whatever dataclass the subclass
            chose for its own use.
        phases : Mapping[str, Phase]
            All known phases, keyed by name.

        Returns
        -------
        tuple[float, _StepResult] | None
            ``(T0, step0)`` to start tracing from, or ``None`` if no
            valid seed could be produced for this candidate.
        """

    @abstractmethod
    def _refine_step(self, cand, phases, T: float,
                     mu_lo: float, mu_hi: float) -> _StepResult:
        """Run one isothermal refinement.

        Parameters
        ----------
        cand
            The propose() candidate.
        phases : Mapping[str, Phase]
            All known phases.
        T : float
            Temperature for this isothermal step.
        mu_lo, mu_hi : float
            Bracket on mu inside which to locate the coexistence.

        Returns
        -------
        _StepResult
            The converged mu* (and any subclass-specific payload).

        Raises
        ------
        ValueError
            To stop the trace at this T (e.g. miscibility-gap closure).
        """

    @abstractmethod
    def _emit(self, cand, T: float, step: _StepResult):
        """Build a :class:`RefinedPoint` or :class:`RefinedMiscibilityGap`
        from one converged step.

        Parameters
        ----------
        cand
            The propose() candidate (carries phase names etc.).
        T : float
            Temperature of the refined transition.
        step : _StepResult
            Output of :meth:`_seed_step` or :meth:`_refine_step`.
        """

    @abstractmethod
    def _pair_key(self, cand):
        """Hashable identifier shared by candidates on the same
        coexistence line, used by :meth:`run` to maintain a per-line
        list of already-traced points for the straddle dedup."""

    def _predict_mu(self, mu_star: float, dmu_dT: float, dT: float) -> float:
        """Predict the next mu* given the current point and slope.

        Default is plain linear extrapolation. Subclasses can override
        to put a fancier predictor here (higher-order, with curvature,
        etc.) without touching the rest of the trace loop.

        Parameters
        ----------
        mu_star : float
            Last converged coexistence mu.
        dmu_dT : float
            Local Clausius-Clapeyron slope from the previous step.
        dT : float
            Signed temperature increment for the next step.
        """
        return mu_star + dmu_dT * dT

    def _dT_adapt(self, step: _StepResult, dmu_dT: float,
                  half_width: float) -> float:
        """Step size for the next predictor step.

        Default keeps the predicted mu drift bounded by ``half_width``:
        ``dT ~ half_width / |dmu/dT|``. Subclasses can override to scale
        on something else (the miscibility-gap refiner uses
        ``dT_max * gap^2`` to tighten naturally near T_c).

        Parameters
        ----------
        step : _StepResult
            Result of the previous refinement; subclasses can read
            ``step.extra`` for their own quantities.
        dmu_dT : float
            Local Clausius-Clapeyron slope.
        half_width : float
            Half the seed simplex's mu bracket width.

        Returns
        -------
        float
            Unsigned step size; the trace loop applies sign and clamps
            to ``[dT_min, dT_max]``.
        """
        # 1e-9 is a tiny floor on the slope so we don't divide by zero
        # when the coexistence line is essentially flat (e.g. a symmetric
        # miscibility gap, where the gap-based override below kicks in).
        slope = max(abs(dmu_dT), 1e-9)
        return half_width / slope

    # -- shared trace skeleton ---------------------------------------------

    def _trace(self, cand, phases, T0, mu0, half_width, T_target, sign):
        """Walk from the seed ``(T0, mu0)`` toward ``T_target``.

        On each step the next mu* is given by :meth:`_predict_mu` and
        corrected by an isothermal :meth:`_refine_step` inside a
        bracket centred on the prediction. The step size comes from
        :meth:`_dT_adapt`, clamped to ``[dT_min, dT_max]``.

        Stops on any of:

        * stepping past ``T_target`` (bounded walk);
        * ``_refine_step`` raising :class:`ValueError` — used both for
          "natural" stops (the miscibility gap closed) and for
          downright failures (no root in the bracket). Either way the
          trace ends silently here and any points produced so far are
          kept;
        * landing on an already-traced segment of the same coexistence
          line (``_point_on_line`` check on ``cand.existing``), to
          avoid retracing.

        Parameters
        ----------
        cand
            The propose() candidate.
        phases : Mapping[str, Phase]
            All known phases.
        T0, mu0 : float
            Seed location.
        half_width : float
            Half-width of the seed simplex's mu bracket; sets the
            scale of subsequent isothermal brackets and the on-line
            tolerance.
        T_target : float
            Walk bound (typically ``cand.T_min`` or ``cand.T_max``).
        sign : int
            ``+1`` for an upward walk in T, ``-1`` for downward.

        Yields
        ------
        RefinedPoint or RefinedMiscibilityGap
            One refined transition per converged step (excluding the
            seed itself, which the caller emits separately).
        """
        dT_boot = sign * min(self.dT_min, abs(T_target - T0))
        if dT_boot == 0:
            return
        T_b = T0 + dT_boot
        try:
            step = self._refine_step(
                cand, phases, T_b, mu0 - half_width, mu0 + half_width)
        except ValueError:
            # Bootstrap step failed. Exercised mainly when the seed
            # already sits at the gap's closure (refine raises
            # because gap < gap_close) or when brentq can't bracket
            # a root one dT_min away from a boundary seed. Both are
            # benign: leave the seed-only point alone, let the
            # straddle dedup downstream decide whether to absorb it.
            return
        yield self._emit(cand, T_b, step)
        mu_star = step.mu_star
        dmu_dT = (mu_star - mu0) / dT_boot
        T = T_b

        for _ in range(self.max_steps):
            # Reached the walk's T-boundary (tolerance just shy of zero
            # so the last step's truncation below doesn't trip us).
            if sign * (T_target - T) <= 1e-12:
                return

            # Adaptive temperature step, clipped to [dT_min, dT_max].
            dT_adapt = self._dT_adapt(step, dmu_dT, half_width)
            dT_adapt = min(self.dT_max, max(self.dT_min, dT_adapt))
            dT = sign * dT_adapt
            # Don't step past the walk boundary; clip dT instead.
            if sign * (T + dT - T_target) > 0:
                dT = T_target - T

            T_next = T + dT
            mu_predicted = self._predict_mu(mu_star, dmu_dT, dT)
            # Corrector bracket: wide enough to absorb the linear
            # extrapolation error (|dmu/dT * dT| * 2) and never
            # narrower than half the seed bracket — too tight a
            # bracket would let the root drift outside.
            bracket = max(half_width * 0.5, abs(dmu_dT * dT) * 2.0)
            try:
                step = self._refine_step(
                    cand, phases, T_next,
                    mu_predicted - bracket, mu_predicted + bracket)
            except ValueError:
                # Trace stops cleanly here — natural closure (gap_close
                # in MiscibilityGapRefiner), or numerical failure of
                # the corrector (e.g. brentq can't bracket a root).
                return
            dmu_dT = (step.mu_star - mu_star) / dT
            T, mu_star = T_next, step.mu_star
            yield self._emit(cand, T, step)
            # If we've walked back onto an already-traced segment of
            # the same coexistence line, stop instead of duplicating it.
            if _point_on_line(T, mu_star, cand.existing, tol_mu=half_width):
                return

    def solve(self, cand, phases):
        seed = self._seed_step(cand, phases)
        if seed is None:
            return []
        T0, step0 = seed
        mu_lo, mu_hi = cand.mu_bracket
        half_width = (mu_hi - mu_lo) / 2.0
        out = [self._emit(cand, T0, step0)]
        out.extend(self._trace(cand, phases, T0, step0.mu_star,
                               half_width, cand.T_max, +1))
        out.extend(self._trace(cand, phases, T0, step0.mu_star,
                               half_width, cand.T_min, -1))
        return out

    def run(self, df: pd.DataFrame, phases: Mapping[str, Phase]) -> pd.DataFrame:
        rows: list[dict] = []
        # Per-pair list of completed traces; each trace is a tuple of
        # (T, mu) points in walk order. Keeping traces separate (rather
        # than flattening into one polyline) is what stops
        # _simplex_straddles inventing fake cross-segments between
        # unrelated single-point seed traces.
        traced: dict[object, list[Trace]] = {}
        # One boundary_id per coexistence line (keyed by _pair_key).
        boundary_ids: dict[object, int] = {}
        next_bid = 0
        for cand in self.propose(df):
            key = self._pair_key(cand)
            traces = tuple(traced.get(key, ()))
            if _simplex_straddles(cand, traces):
                continue
            cand = replace(cand, existing=traces)
            pts = list(self.solve(cand, phases))
            if pts:
                new_trace = tuple(
                    sorted(((p.T, p.mu) for p in pts), key=lambda tm: tm[0]))
                traced.setdefault(key, []).append(new_trace)
            if key not in boundary_ids:
                boundary_ids[key] = next_bid
                next_bid += 1
            bid = boundary_ids[key]
            for pt in pts:
                if pt.T < 0 or _dominated(pt, phases):
                    continue
                rows.extend(replace(pt, boundary_id=bid).to_rows(phases))
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["stable"] = True
        out["border"] = True
        out["refined"] = self.label
        return out


# -- Inter-phase coexistence --------------------------------------------------


@dataclass(frozen=True)
class _InterCandidate:
    """Seed for :class:`ClausiusClapeyronRefiner` — inter-phase boundary.

    ``proj_p1`` and ``proj_p2`` are the in-simplex (T, mu) vertex
    centroids of each phase. The line between them is guaranteed to
    cross the boundary, so brentq always has a bracket for the seed.
    """

    phase1: str
    phase2: str
    T_seed: float
    mu_bracket: Bracket
    T_bracket: Bracket
    T_min: float
    T_max: float
    proj_p1: TMuPoint
    proj_p2: TMuPoint
    existing: TraceList = field(default=())


class ClausiusClapeyronRefiner(_CCBase):
    """Predictor-corrector tracer of two-phase coexistence lines.

    For each two-phase Delaunay simplex, projects across the line
    between the two phases' vertex centroids to land a seed point on
    the boundary, then walks T in both directions via isothermal
    root-finding on ``phi1 - phi2``. The default ``_dT_adapt`` keeps
    each step's mu drift bounded by the seed bracket's half-width.

    Parameters
    ----------
    dT_max : float
        Maximum allowed temperature step (K). See :class:`_CCBase`.
    dT_min : float
        Minimum temperature step / bootstrap step (K). See
        :class:`_CCBase`.
    max_steps : int
        Hard cap on steps per trace direction. See :class:`_CCBase`.
    """

    label = "clausius-clapeyron"

    def propose(self, df: pd.DataFrame) -> Iterator[_InterCandidate]:
        T_min = float(df["T"].min())
        T_max = float(df["T"].max())
        for simplex, n in _delaunay_simplices(df):
            if n != 2:
                continue
            mu_lo, mu_hi, T_lo, T_hi, T_seed = _simplex_brackets(simplex)
            if mu_hi <= mu_lo:
                continue
            name1, name2 = simplex.phase.unique()
            p1xy, p2xy = _phase_centroids_xy(simplex)
            yield _InterCandidate(
                phase1=name1, phase2=name2,
                T_seed=T_seed,
                mu_bracket=(mu_lo, mu_hi),
                T_bracket=(T_lo, T_hi),
                T_min=T_min, T_max=T_max,
                proj_p1=p1xy, proj_p2=p2xy,
            )

    def _pair_key(self, cand):
        return frozenset((cand.phase1, cand.phase2))

    def _seed_step(self, cand, phases):
        p1xy = np.array(cand.proj_p1)
        p2xy = np.array(cand.proj_p2)
        p1, p2 = phases[cand.phase1], phases[cand.phase2]

        def project(t):
            T, mu = p1xy + (p2xy - p1xy) * t
            return float(T), float(mu)

        try:
            t = _find_one_point(
                p1, p2,
                lambda p, x: p.semigrand_potential(*project(x)),
                (0.0, 1.0),
            )
        except ValueError:
            warnings.warn(
                f"CC refiner: projection failed for "
                f"{cand.phase1}/{cand.phase2} at simplex T~{cand.T_seed:.2f}",
                stacklevel=2,
            )
            return None
        T0, mu0 = project(t)
        return T0, _StepResult(mu_star=mu0)

    def _refine_step(self, cand, phases, T, mu_lo, mu_hi):
        p1, p2 = phases[cand.phase1], phases[cand.phase2]
        mu = _find_one_point(
            p1, p2,
            lambda p, x: p.semigrand_potential(T, x),
            (mu_lo, mu_hi),
        )
        return _StepResult(mu_star=mu)

    def _emit(self, cand, T, step):
        return RefinedPoint(
            T=T, mu=step.mu_star, phases=(cand.phase1, cand.phase2))


# -- Intra-phase miscibility gap ----------------------------------------------


@dataclass(frozen=True)
class _GapStep:
    """Payload of one miscibility-gap refinement step."""

    c_left: float
    c_right: float
    gap: float


@dataclass(frozen=True)
class _GapCandidate:
    """Seed for :class:`MiscibilityGapRefiner` — intra-phase gap."""

    phase: str
    T_seed: float
    mu_bracket: Bracket
    T_bracket: Bracket
    T_min: float
    T_max: float
    existing: TraceList = field(default=())


class MiscibilityGapRefiner(_CCBase):
    """Predictor-corrector tracer of intra-phase miscibility gaps.

    For each single-phase simplex with a wide-enough c-spread, scans
    c(mu) on the simplex's mu bracket and locates the steepest-dc/dmu
    segment — the binodal jump. The trace walks T in both directions
    with step size ``dT_max * gap^2`` (gap shrinks naturally toward T_c).

    Stops when:

    * the gap shrinks below ``gap_close`` (essentially closed), or
    * the dominant scan segment carries less than ``gap_share_min`` of
      the total c-variation, signalling a smooth supercritical c(mu).

    Each transition emits a :class:`RefinedMiscibilityGap` carrying
    the steepest-segment endpoints so its :meth:`~RefinedMiscibilityGap.to_rows`
    reads concentrations on each side of the jump.

    Parameters
    ----------
    dT_max, dT_min, max_steps :
        See :class:`_CCBase`.
    c_jump_min : float
        Minimum c-spread of a simplex's vertices for it to be seeded.
    gap_close : float
        Trace-time absolute c-jump threshold (default 1e-3).
    gap_share_min : float
        Dominant-segment share of total scan c-variation below which
        the bracket is judged supercritical (default 0.1). Raise
        toward 0.5 to stop the trace further below T_c; lower toward
        0 to push closer to T_c at the cost of some drift.
    """

    label = "miscibility-gap"

    def __init__(self, dT_max: float = 50.0, dT_min: float = 1.0,
                 max_steps: int = 500, c_jump_min: float = 0.3,
                 gap_close: float = 1e-3, gap_share_min: float = 0.1):
        super().__init__(dT_max=dT_max, dT_min=dT_min, max_steps=max_steps)
        self.c_jump_min = c_jump_min
        self.gap_close = gap_close
        self.gap_share_min = gap_share_min

    def propose(self, df: pd.DataFrame) -> Iterator[_GapCandidate]:
        T_min = float(df["T"].min())
        T_max = float(df["T"].max())
        # Yield widest-c-spread simplex first: the most reliable
        # simplex seeds the trace, and neighbouring ones get straddle-
        # skipped on subsequent iterations.
        candidates: list[tuple[float, _GapCandidate]] = []
        for simplex, n in _delaunay_simplices(df):
            if n != 1:
                continue
            mu_lo, mu_hi, T_lo, T_hi, T_seed = _simplex_brackets(simplex)
            if mu_hi <= mu_lo:
                continue
            c_spread = float(simplex["c"].max() - simplex["c"].min())
            if c_spread < self.c_jump_min:
                continue
            candidates.append((-c_spread, _GapCandidate(
                phase=simplex.phase.iloc[0],
                T_seed=T_seed,
                mu_bracket=(mu_lo, mu_hi),
                T_bracket=(T_lo, T_hi),
                T_min=T_min, T_max=T_max,
            )))
        candidates.sort(key=lambda kv: kv[0])
        for _, cand in candidates:
            yield cand

    def _pair_key(self, cand):
        return cand.phase

    def _scan(self, ph, T, mu_lo, mu_hi) -> _StepResult:
        """Dense c(mu) scan picking the steepest-dc/dmu segment."""
        mus = np.linspace(mu_lo, mu_hi, 50)
        cs = np.array([float(ph.concentration(T, m)) for m in mus])
        dcs = np.diff(cs)
        if dcs.max() <= 0:
            raise ValueError(
                f"c(mu) is not increasing in [{mu_lo}, {mu_hi}] at T={T}")
        i = int(np.argmax(dcs))
        mu_l, mu_r = float(mus[i]), float(mus[i + 1])
        c_l, c_r = float(cs[i]), float(cs[i + 1])
        total = float(cs.max() - cs.min())
        gap = c_r - c_l
        # Concentration ratio: at a sharp binodal the dominant segment
        # holds essentially all of the c variation in the bracket; for
        # a smooth supercritical c(mu) it spreads across the scan and
        # this ratio drops toward 1/49.
        if total > 0 and gap / total < self.gap_share_min:
            gap = 0.0
        return _StepResult(
            mu_star=(mu_l + mu_r) / 2.0,
            extra=_GapStep(c_left=c_l, c_right=c_r, gap=gap),
        )

    def _seed_step(self, cand, phases):
        try:
            step = self._scan(phases[cand.phase], cand.T_seed, *cand.mu_bracket)
        except ValueError:
            warnings.warn(
                f"miscibility-gap refiner: no jump for {cand.phase} at "
                f"T={cand.T_seed:.2f} in mu={cand.mu_bracket}",
                stacklevel=2,
            )
            return None
        return cand.T_seed, step

    def _refine_step(self, cand, phases, T, mu_lo, mu_hi):
        step = self._scan(phases[cand.phase], T, mu_lo, mu_hi)
        if step.extra.gap < self.gap_close:
            raise ValueError("miscibility gap closed")
        return step

    def _dT_adapt(self, step, dmu_dT, half_width):
        # Step shrinks naturally as the gap closes near T_c.
        return self.dT_max * step.extra.gap ** 2

    def _emit(self, cand, T, step):
        x = step.extra
        return RefinedMiscibilityGap(
            T=T, mu=step.mu_star, phase=cand.phase,
            c_left=x.c_left, c_right=x.c_right)


# -- Defaults -----------------------------------------------------------------


def default_refiners(df: pd.DataFrame) -> Sequence[Refiner]:
    """Pick a sensible default refiner list for ``df``.

    * Both ``mu`` and ``T`` sampled (2-D grid) → triple points
      (:class:`DelaunayTripleRefiner`) plus dense Clausius-Clapeyron
      traces for inter-phase boundaries and intra-phase miscibility
      gaps.
    * Only one axis sampled (1-D scan) → just the
      :class:`ScanRefiner` walking that axis.
    """
    multiple_mus = len(df["mu"].unique()) > 1
    multiple_ts = len(df["T"].unique()) > 1
    if multiple_mus and multiple_ts:
        return [
            DelaunayTripleRefiner(),
            ClausiusClapeyronRefiner(),
            MiscibilityGapRefiner(),
        ]
    refiners: list[Refiner] = []
    if multiple_mus:
        refiners.append(ScanRefiner(by="mu"))
    if multiple_ts:
        refiners.append(ScanRefiner(by="T"))
    return refiners
