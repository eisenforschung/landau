"""
Calculates phase diagrams from sets of Phases.
"""

import numbers
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann, eV
from sklearn.cluster import AgglomerativeClustering


from .features import Locus
from .phases import Phase, AbstractLinePhase
from .refine import Refiner, ScanRefiner, default_refiners, _find_one_point as find_one_point  # noqa: F401


kB = Boltzmann / eV


__all__ = ["calc_phase_diagram", "get_transitions", "cluster_T_c", "cluster_T_c_mu"]


_PHASE_UNIT_SEP = "_"


def _apply_series(grouped, fn, name: str) -> pd.Series:
    """Run ``fn`` per group, return a row-indexed :class:`pd.Series`.

    Wraps :meth:`pandas.core.groupby.DataFrameGroupBy.apply` so that the result
    has the same shape across pandas 2 and 3.  In pandas 3 a single-group
    ``groupby.apply`` whose callable returns a Series collapses to a one-row
    DataFrame indexed by the group key; returning a DataFrame from the callable
    is the documented workaround that combines consistently across both
    versions (pandas user guide, "Flexible apply").  ``include_groups=False``
    is passed through.
    """
    return grouped.apply(lambda g: fn(g).to_frame(name), include_groups=False)[name]


def _join_phase_unit(phase: pd.Series, unit: pd.Series) -> pd.Series:
    """Encode ``(phase, unit)`` pairs as ``f"{phase}{_PHASE_UNIT_SEP}{unit}"`` strings.

    Inverse is :func:`_split_phase_unit`. The two functions round-trip as long as
    the unit-side string is a valid integer literal — the split uses ``rsplit``
    with ``n=1``, so underscores inside ``phase`` are preserved.
    """
    return phase.astype(str) + _PHASE_UNIT_SEP + unit.astype(str)


def _split_phase_unit(combined: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inverse of :func:`_join_phase_unit`. Returns ``(phase, unit)``."""
    parts = combined.str.rsplit(_PHASE_UNIT_SEP, n=1)
    phase = parts.map(lambda x: x[0])
    unit = parts.map(lambda x: int(x[1]))
    return phase, unit


# How far past the sampled mu range, in units of kB*T at the highest temperature,
# the pure-component free energies are read off; a phase that reaches a pure
# component sits within ~exp(-40) of it there.
_TERMINAL_KT_FAR = 40.0
# How close to c = 0 / 1 a phase must sit at that mu to count as reaching it.
_TERMINAL_C_TOL = 1e-6


def _terminal_phases(phases, grid_Ts, mu: float, end: int) -> pd.DataFrame | None:
    """Stable phase at the pure component ``end`` (``0`` or ``1``) along ``T`` at fixed ``mu``.

    Samples every phase on ``grid_Ts`` at ``mu``, keeps those whose
    concentration is within ``_TERMINAL_C_TOL`` of ``end``, takes the lowest
    ``phi`` per temperature and locates the transitions between them with
    :class:`~landau.refine.ScanRefiner` along ``T``.

    Returns:
        frame with columns ``T`` and ``phase`` sorted by ``T`` (a transition
        appears once per phase meeting there), or ``None`` if at some grid
        temperature no phase reaches ``end``.
    """
    rows = []
    for T in grid_Ts:
        for p in phases.values():
            c = float(p.concentration(T, mu))
            if abs(c - end) < _TERMINAL_C_TOL:
                rows.append({"T": T, "mu": mu, "phase": p.name, "phi": float(p.semigrand_potential(T, mu)), "c": c})
    sampled = pd.DataFrame(rows, columns=["T", "mu", "phase", "phi", "c"])
    if sampled["T"].nunique() < len(grid_Ts):
        return None
    stable = sampled.loc[sampled.groupby("T").phi.idxmin()].reset_index(drop=True)
    stable["stable"] = True
    if len(grid_Ts) > 1:
        reaching = {name: phases[name] for name in sampled["phase"].unique()}
        transitions = ScanRefiner(by="T").run(stable, reaching)
        if not transitions.empty:
            stable = pd.concat([stable[["T", "phase"]], transitions[["T", "phase"]]], ignore_index=True)
    return stable[["T", "phase"]].sort_values("T", kind="stable").reset_index(drop=True)


def _terminal_free_energies(phases, grid_Ts, Ts, mu_lo: float, mu_hi: float) -> np.ndarray | None:
    """Lowest pure-component free energies ``f(T, c=0)`` and ``f(T, c=1)`` at temperatures ``Ts``.

    The stable phase at each pure component is resolved once along ``grid_Ts``
    (:func:`_terminal_phases`), ``_TERMINAL_KT_FAR * kB * max(T)`` below
    ``mu_lo`` and above ``mu_hi``.  At each of ``Ts`` only the phase stable
    there in that sequence is evaluated: ``phi`` at ``c = 0``, ``phi + mu`` at
    ``c = 1``.  Temperatures outside ``grid_Ts`` take the phase at the nearer
    end of it.

    Args:
        phases: mapping of phase name to :class:`Phase`
        grid_Ts: temperatures the phase diagram was sampled at
        Ts: temperatures to evaluate the pure-component free energies at
        mu_lo, mu_hi: sampled chemical potential window

    Returns:
        array of shape ``(len(Ts), 2)`` holding ``f(T, 0)`` and ``f(T, 1)``,
        or ``None`` if at some grid temperature no phase reaches one of the
        pure components.
    """
    grid_Ts = np.unique(np.asarray(grid_Ts, dtype=float))
    Ts = np.asarray(Ts, dtype=float)
    far = _TERMINAL_KT_FAR * kB * max(np.abs(grid_Ts).max(), np.abs(Ts).max())
    out = np.empty((len(Ts), 2))
    for end, mu in ((0, mu_lo - far), (1, mu_hi + far)):
        sequence = _terminal_phases(phases, grid_Ts, mu, end)
        if sequence is None:
            return None
        seq_T = sequence["T"].to_numpy()
        seq_phase = sequence["phase"].to_numpy()
        stable = seq_phase[np.minimum(np.searchsorted(seq_T, Ts), len(seq_T) - 1)]
        for i, (T, name) in enumerate(zip(Ts, stable)):
            out[i, end] = float(phases[name].semigrand_potential(T, mu)) + end * mu
    return out


def _f_excess(df: pd.DataFrame, phases, grid_Ts) -> pd.Series | None:
    """Free energy of each row relative to the pure components at its temperature.

    ``f_excess = f - [f0 * (1 - c) + f1 * c]``, where ``f0`` and ``f1`` are
    the lowest free energies of any phase at ``c = 0`` and ``c = 1`` at the
    row's ``T`` (:func:`_terminal_free_energies`), evaluated from ``phases``
    rather than read off the sampled rows, so they do not depend on the
    sampling window or on which rows share a temperature.

    Args:
        df: frame with columns ``T``, ``mu``, ``c`` and ``f``
        phases: mapping of phase name to :class:`Phase`
        grid_Ts: temperatures the phase diagram was sampled at

    Returns:
        series aligned to ``df.index`` with ``NaN`` on rows with
        ``mu = +-inf``, or ``None`` if the phases do not reach both pure
        components at every grid temperature (the reference is undefined).
    """
    finite = df.query("-inf<mu<inf")
    if finite.empty:
        return None
    Ts = finite["T"].unique()
    ref = _terminal_free_energies(phases, grid_Ts, Ts, finite["mu"].min(), finite["mu"].max())
    if ref is None:
        return None
    f0 = finite["T"].map(dict(zip(Ts, ref[:, 0])))
    f1 = finite["T"].map(dict(zip(Ts, ref[:, 1])))
    return (finite.f - (f0 * (1 - finite.c) + f1 * finite.c)).reindex(df.index)


def _split_stable(df):
    udf = df.query("not stable").reset_index(drop=True)
    udf["border"] = False
    udf["locus"] = Locus.INTERIOR
    sdf = df.query("stable").reset_index(drop=True)
    sdf["border"] = False
    sdf["refined"] = "no"
    sdf["locus"] = Locus.INTERIOR
    return sdf, udf


def _border_edges(df, min_c, max_c):
    """Mark T extremes as border and emit synthetic +-inf mu edges (2D only).

    The synthetic edge points only close the sampling window for polygon
    construction, they are no thermodynamic transitions, so they keep
    ``locus = Locus.INTERIOR`` despite ``border = True``.
    """
    df.loc[df["T"] == df["T"].min(), "border"] = True
    df.loc[df["T"] == df["T"].max(), "border"] = True
    left = df.loc[df["mu"] == df["mu"].min()][["phase", "T"]].copy()
    left["mu"] = -np.inf
    left["c"] = min_c
    left["border"] = True
    left["stable"] = True
    left["locus"] = Locus.INTERIOR
    right = df.loc[df["mu"] == df["mu"].max()][["phase", "T"]].copy()
    right["mu"] = +np.inf
    right["c"] = max_c
    right["border"] = True
    right["stable"] = True
    right["locus"] = Locus.INTERIOR
    return left, right


def refine_phase_diagram(
    df: pd.DataFrame,
    phases,
    min_c: float = 0,
    max_c: float = 1,
    refiners: Sequence[Refiner] | None = None,
) -> pd.DataFrame:
    """Add additional points to a coarse phase diagram by searching for exact transitions.

    Args:
        df: dataframe of sampled phase points (with ``stable`` column)
        phases: mapping of phase name to :class:`Phase`
        min_c, max_c: concentration bounds used for the synthetic ``mu=+-inf`` border points
        refiners: sequence of :class:`landau.refine.Refiner` to apply. If
            ``None``, picks defaults based on which of ``T``/``mu`` are sampled.
    """
    sdf, udf = _split_stable(df)
    data = [sdf, udf]
    multiple_mus = len(sdf["mu"].unique()) > 1
    multiple_ts = len(sdf["T"].unique()) > 1
    if multiple_mus and multiple_ts:
        data.extend(_border_edges(sdf, min_c, max_c))
    if refiners is None:
        refiners = default_refiners(sdf)
    for r in refiners:
        out = r.run(sdf, phases)
        if not out.empty:
            data.append(out)
    return pd.concat(data, ignore_index=True)


def _semigrand_average_concentration(phases: Iterable[Phase], T: float, mu):
    """Boltzmann-weighted mean concentration across ``phases`` at ``(T, mu)``.

    A smooth, phase-agnostic stand-in for the equilibrium concentration: every
    phase contributes its concentration weighted by ``exp(-phi / kB T)``, so the
    result varies continuously across phase boundaries instead of jumping with
    the ``argmin`` phase.  :func:`guess_mu_range` inverts this c(mu) mapping.
    ``mu`` may be a scalar or an array.
    """
    phis = np.array([p.semigrand_potential(T, mu) for p in phases])
    conc = np.array([p.concentration(T, mu) for p in phases])
    phis = phis - phis.min(axis=0)
    prob = np.exp(-phis / (kB * T))
    prob /= prob.sum(axis=0)
    return (prob * conc).sum(axis=0)


def guess_mu_range(phases: Iterable[Phase], T: float, samples: int, tolerance: float = 1e-2):
    """Guess chemical potential window from the ideal solution.

    Searches numerically for chemical potentials which stabilize
    concentrations close to 0 and 1 and then use the concentrations
    encountered along the way to numerically invert the c(mu) mapping.
    Using an even c grid with mu(c) then yields a decent sampling of mu
    space so that the final phase diagram is described everywhere equally.

    Args:
        phases: list of phases to consider
        T: temperature at which to estimate mu(c)
        samples: how many mu samples to return

    Returns:
        array of chemical potentials that likely cover the whole concentration space
    """
    # TODO: this can be used immediately also for the actual phase diagram
    # calculation: keep track of which phase is the most likely
    import scipy.optimize as so
    import scipy.interpolate as si

    def c(mu):
        return _semigrand_average_concentration(phases, T, mu)

    # finish=None keeps the result on the scanned grid.  The default fmin polish
    # is unconstrained and, because c(mu) approaches 0 and 1 only asymptotically
    # along the single-phase tails, walks far outside the (-10, 10) bracket.  That
    # blows the mm grid spacing up and pushes the inverted mu window deep into the
    # single-phase regions.
    mu0 = float(np.atleast_1d(so.brute(lambda x: +c(x[0]), ranges=[(-10, 10)], Ns=200, finish=None))[0])
    mu1 = float(np.atleast_1d(so.brute(lambda x: -c(x[0]), ranges=[(-10, 10)], Ns=200, finish=None))[0])
    if mu0 == mu1:
        if tolerance > 1e-7:
            return guess_mu_range(phases, T, samples, tolerance/10)
        raise ValueError(
                "chemical potential range degenerate! Check that phases that not all phases have the same fixed "
                "concentration!"
        )
    mm = np.linspace(mu0, mu1, samples)
    cc = c(mm)
    # When phases sit close together in concentration (e.g. two line phases a few
    # at% apart), the c(mu) span reachable within the (-10, 10) bracket can be
    # narrower than 2*tolerance, so c0 = min(cc) + tolerance would cross past
    # c1 = max(cc) - tolerance and land outside the sampled range entirely.
    # Shrink tolerance the same way the mu0 == mu1 degeneracy above does until
    # the window fits inside the reachable span.
    if max(cc) - min(cc) < 2 * tolerance:
        if tolerance > 1e-7:
            return guess_mu_range(phases, T, samples, tolerance / 10)
        raise ValueError(
                "chemical potential range degenerate! Concentration span reachable within the "
                "sampled bracket is narrower than 2*tolerance even at tolerance=1e-7 - check that "
                "phases are not extremely close in concentration."
        )
    c0 = min(cc) + tolerance
    c1 = max(cc) - tolerance
    # At very low T, c(mu) is a step function and cc may have repeated values;
    # sort and deduplicate before passing to interp1d which requires monotone x.
    sort_idx = np.argsort(cc)
    cc_s, mm_s = cc[sort_idx], mm[sort_idx]
    _, unique_idx = np.unique(cc_s, return_index=True)
    return si.interp1d(cc_s[unique_idx], mm_s[unique_idx])(np.linspace(c0, c1, samples)), c0, c1


def calc_phase_diagram(
    phases: Iterable[Phase],
    Ts: Iterable[float] | float,
    mu: Iterable[float] | float | int,
    refine: bool | Sequence[Refiner] = True,
    keep_unstable: bool = False,
):
    """
    Calculate phase diagram at given sampling points.

    Args:
        phases (iterable of Phases)
        Ts (iterable of floats): sampling points in temperature
        mu (iterable of floats): sampling points in chemical potential; if a
            non-zero int, guess sampling points with guess_mu_range at max(Ts)
            (pass ``mu=0.0`` for a fixed-mu temperature-only diagram)
        refine (bool or sequence of :class:`~landau.refine.Refiner`): add
            additional sampling points at exact phase transitions. ``True``
            picks defaults based on which of ``T``/``mu`` are sampled, ``False``
            skips refinement, and an explicit sequence of
            :class:`~landau.refine.Refiner` instances applies just those.
        keep_unstable (bool): if ``True``, keep entries of all phases at all sampling points, otherwise only keep entries of stable phases

    Returns:
        dataframe of phase points; the ``locus`` column classifies each row
        as a :class:`~landau.features.Locus` value (``"interior"``,
        ``"boundary"``, ``"triple"`` or ``"congruent"``); with
        ``keep_unstable=True`` the ``dphi`` column is ``phi`` minus the
        lowest ``phi`` among the rows at the same ``(T, mu)``, i.e. how far
        a phase sits above the stable one (``0`` for stable rows, ``NaN`` on
        the synthetic ``mu = +-inf`` edges; refined rows are zero up to the
        refiners' numerical tolerance). The ``f_excess`` column is ``f``
        relative to the chord between the lowest free energies of the pure
        components at each row's ``T``; it is left out when the phases do not
        reach both ``c = 0`` and ``c = 1``.
    """
    if not isinstance(Ts, Iterable):
        Ts = [Ts]
    phases = {p.name: p for p in phases}
    if isinstance(mu, numbers.Integral) and mu != 0:
        try:
            mu, *_ = guess_mu_range(phases.values(), max(Ts), int(mu))
        except ValueError:
            if all(isinstance(p, AbstractLinePhase) for p in phases.values()):
                raise ValueError(
                        "Cannot guess chemical potential range of line phases with all the same concentration!"
                ) from None
            raise

    def get(s, T):
        phi = s.semigrand_potential(T, mu)
        return {"T": T, "phase": s.name, "phi": phi, "mu": mu, "c": s.concentration(T, mu)}

    pdf = pd.DataFrame([get(s, T) for s in phases.values() for T in Ts])
    pdf = pdf.explode(["mu", "phi", "c"]).infer_objects().reset_index(drop=True)
    pdf["stable"] = False
    pdf.loc[pdf.groupby(["T", "mu"], group_keys=False).phi.idxmin(), "stable"] = True
    pdf["locus"] = Locus.INTERIOR
    if refine is not False:
        refiners = None if isinstance(refine, bool) else refine
        min_c = pdf.c.min()
        max_c = pdf.c.max()
        pdf = refine_phase_diagram(pdf, phases, min_c=min_c, max_c=max_c, refiners=refiners)
    pdf["f"] = pdf.phi + pdf.mu * pdf.c
    f_excess = _f_excess(pdf, phases, Ts)
    if f_excess is not None:
        pdf["f_excess"] = f_excess
    if keep_unstable:
        pdf["dphi"] = pdf.phi - pdf.groupby(["T", "mu"]).phi.transform("min")
    else:
        pdf = pdf.query("stable")
    return pdf


def _tie_lines(dd):
    """Pair the border rows at one ``(mu, T)`` into two-phase tie-lines.

    ``dd`` must hold only the ``border`` rows of a single ``(mu, T)``, sorted
    by ``c``, with columns ``c`` and ``phase``. Neighbouring rows form a
    tie-line named ``"A-B"`` in ascending ``c``, so ``n`` rows give ``n - 1``
    tie-lines and a row between two others is emitted once per tie-line. A
    single row (the ``mu = +-inf`` edges) is emitted under its own phase name.

    Returns one row per tie-line end, with columns ``transition``, ``c`` and
    ``phase``.
    """
    c, phase = dd["c"].to_numpy(), dd["phase"].to_numpy()
    if len(dd) == 1:
        return pd.DataFrame({"transition": [phase[0]], "c": c, "phase": phase})
    rows = []
    for i in range(len(dd) - 1):
        name = f"{phase[i]}-{phase[i + 1]}"
        rows.append((name, c[i], phase[i]))
        rows.append((name, c[i + 1], phase[i + 1]))
    return pd.DataFrame(rows, columns=["transition", "c", "phase"])


def _rescale_T(t):
    tmin, tmax = t.min(), t.max()
    if tmin != tmax:
        return (t - tmin) / (tmax - tmin)
    return t


def _agglomerative_clusterer(distance_threshold):
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="single",
    )


def cluster_T_c(dd, *, distance_threshold) -> pd.Series:
    """Cluster points by (T, c) coordinates; used by cluster_phase.

    Args:
        dd: DataFrame with columns 'T' and 'c'.
        distance_threshold: Required. Agglomerative clustering cut-off in normalised
            (T, c) space. Smaller values split more aggressively; larger values merge
            more. Values of roughly 0.01-0.3 keep the test fixtures' disconnected
            same-phase fields apart (see ``tests/regression/test_issue_456.py``), while
            over-splitting of connected fields starts between 0.05 and 0.1. The
            user-facing default lives on :func:`~landau.plot.cluster_phase` and
            :func:`~landau.plot.get_polygons`.
    """
    if dd.empty:
        return pd.Series(dtype=np.intp)
    t = _rescale_T(dd["T"])
    ids = _agglomerative_clusterer(distance_threshold).fit_predict(np.transpose([t, dd["c"]]))
    return pd.Series(ids, index=dd.index)


def cluster_T_c_mu(dd, *, distance_threshold) -> pd.Series:
    """Cluster points by (T, c, mu); rows with mu=±inf get their own distinct labels.

    Used by get_transitions. The mu=±inf rows are border edges from _border_edges
    and must not be fed to AgglomerativeClustering.

    Args:
        dd: DataFrame with columns 'T', 'c', and 'mu'.
        distance_threshold: Agglomerative clustering cut-off in normalised (T, c, mu) space.
            See :func:`cluster_T_c` for tuning guidance.
    """
    if dd.empty:
        return pd.Series(dtype=np.intp)
    t = _rescale_T(dd["T"])
    ids = pd.Series(np.zeros(len(dd), dtype=np.intp), index=dd.index)
    F = np.isfinite(dd["mu"])
    if F.any() and F.sum() >= 2:
        ids.loc[F] = _agglomerative_clusterer(distance_threshold).fit_predict(
            np.transpose([t.loc[F], dd["c"].loc[F], dd["mu"].loc[F]])
        )
    m = ids.max()
    ids.loc[dd["mu"] == +np.inf] = m + 1
    ids.loc[dd["mu"] == -np.inf] = m + 2
    return ids


def cluster(dd, use_mu=True, *, distance_threshold):
    """Thin dispatch to cluster_T_c or cluster_T_c_mu; prefer calling those directly."""
    if use_mu:
        return cluster_T_c_mu(dd, distance_threshold=distance_threshold)
    else:
        return cluster_T_c(dd, distance_threshold=distance_threshold)


def get_transitions(df):
    """
    Identify "continuous" two-phase transition lines in mu/T space, i.e. transitions between the same two phases and along which mu/T are continuous.

    Useful for plotting below, but potentially also to augment the existing refining routines and
    acquire additional Free energies from calphy/etc. to improve the diagram.
    """
    bdf = df.query("border").sort_values("c")
    # the border points at one (mu, T) are the ends of tie-lines: pair them up
    # into two-phase transitions (a lone point at mu=+-inf is its own) and add
    # the 'transition' column; groupby keeps the c order within each group
    tdf = bdf.groupby(["mu", "T"])[["c", "phase"]].apply(_tie_lines, include_groups=False)
    tdf = tdf.reset_index(level=["mu", "T"]).reset_index(drop=True)

    # cluster points that are assigned as one transition, because the same transition can appear multiple times in "disconnected" manner in a phase
    # diagram, e.g. a solid solution in contact with the melt interrupted by a higher melting intermetallic
    if not tdf.empty:
        tdf["transition_unit"] = _apply_series(
            tdf.groupby("transition", group_keys=False),
            lambda g: cluster_T_c_mu(g, distance_threshold=0.5),  # 0.5 hand-tuned
            "transition_unit",
        )
        tdf["border_segment"] = _join_phase_unit(tdf["transition"], tdf["transition_unit"])
    else:
        tdf["transition_unit"] = pd.Series(dtype=np.intp)
        tdf["border_segment"] = pd.Series(dtype=object)

    return tdf
