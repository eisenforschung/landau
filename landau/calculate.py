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
from .refine import Refiner, default_refiners, _find_one_point as find_one_point  # noqa: F401


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


def _f_excess_tangent_chord(dd: pd.DataFrame) -> pd.Series:
    """Subtract from ``dd.f`` the chord between the endpoint reference tangents.

    The reference for each endpoint (``c = 0`` and ``c = 1``) is the smallest
    tangent value among phases whose sampled c-range reaches within 1% of the
    span of that extreme — ``phi`` at ``c = 0``, ``phi + mu`` at ``c = 1``.
    Comparing by the tangent value, rather than by ``f`` at the extreme sample,
    keeps a phase sampled only up to ``c = 1 - eps`` from stealing the reference
    from a line phase sitting exactly at ``c = 1`` (``f`` is convex with slope
    ``mu`` in ``c``, so the partial-coverage phase sits below its own tangent
    value at the endpoint).  For a line phase located exactly at an endpoint
    both expressions reduce to ``f``.  Falls back to the raw ``f`` at the
    extreme sample when no phase reaches the endpoint window.

    Drops ``mu = +-inf`` rows before computing.  Designed to be applied per
    temperature via :func:`_apply_series`.
    """
    dd = dd.query("-inf<mu<inf")
    if dd.empty:
        return dd.f
    c_min = dd.c.min()
    c_max = dd.c.max()
    c_span = c_max - c_min
    lo_thr = c_min + 0.01 * c_span
    hi_thr = c_max - 0.01 * c_span
    f0, f1 = np.inf, np.inf
    tangent0 = dd["phi"]
    tangent1 = dd["phi"] + dd["mu"]
    for _, g in dd.groupby("phase"):
        if g["c"].min() <= lo_thr:
            f0 = min(f0, tangent0[g["c"].idxmin()])
        if g["c"].max() >= hi_thr:
            f1 = min(f1, tangent1[g["c"].idxmax()])
    if not np.isfinite(f0):
        f0 = dd.loc[dd["c"].idxmin(), "f"]
    if not np.isfinite(f1):
        f1 = dd.loc[dd["c"].idxmax(), "f"]
    return dd.f - (f0 * (1 - dd.c) + f1 * dd.c)


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
    refine: bool = True,
    keep_unstable: bool = False,
):
    """
    Calculate phase diagram at given sampling points.

    Args:
        phases (iterable of Phases)
        Ts (iterable of floats): sampling points in temperature
        mu (iterable of floats): sampling points in chemical potential; if int
            guess sampling points with guess_mu_range at max(Ts)
        refine (bool): add additional sampling points at exact phase transitions
        keep_unstable (bool): only keep entries of stable phases, otherwise keep entries of all phases at all sampling points

    Returns:
        dataframe of phase points; the ``locus`` column classifies each row
        as a :class:`~landau.features.Locus` value (``"interior"``,
        ``"boundary"`` or ``"triple"``)
    """
    if not isinstance(Ts, Iterable):
        Ts = [Ts]
    phases = {p.name: p for p in phases}
    if isinstance(mu, numbers.Integral) and mu != 0:
        # we would often pass mu=0 to calculate a fixed mu, temperature only diagram and it'd be a bit annoying to pass
        # mu=0.0 all the time, so we special case as above
        try:
            mu, min_c, max_c = guess_mu_range(phases.values(), max(Ts), int(mu))
        except ValueError:
            if all(isinstance(p, AbstractLinePhase) for p in phases.values()):
                raise ValueError(
                        "Cannot guess chemical potential range of line phases with all the same concentration!"
                ) from None
            raise
    elif refine:
        min_c, max_c = None, None

    def get(s, T):
        phi = s.semigrand_potential(T, mu)
        return {"T": T, "phase": s.name, "phi": phi, "mu": mu, "c": s.concentration(T, mu)}

    pdf = pd.DataFrame([get(s, T) for s in phases.values() for T in Ts])
    pdf = pdf.explode(["mu", "phi", "c"]).infer_objects().reset_index(drop=True)
    pdf["stable"] = False
    pdf.loc[pdf.groupby(["T", "mu"], group_keys=False).phi.idxmin(), "stable"] = True
    pdf["locus"] = Locus.INTERIOR
    if refine:
        min_c = pdf.c.min()
        max_c = pdf.c.max()
        pdf = refine_phase_diagram(pdf, phases, min_c=min_c, max_c=max_c)
    pdf["f"] = pdf.phi + pdf.mu * pdf.c
    pdf["f_excess"] = _apply_series(
        pdf.groupby("T", group_keys=False), _f_excess_tangent_chord, "f_excess"
    )
    if not keep_unstable:
        pdf = pdf.query("stable")
    return pdf


def reduce(dd):
    dd = dd.sort_values("c")
    return pd.Series(
        {
            "transition": "-".join(dd.phase.tolist()),
            "c": dd.c.tolist(),
            "phase": dd.phase.tolist(),
        }
    )


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
        distance_threshold: Agglomerative clustering cut-off in normalised (T, c) space.
            Smaller values split more aggressively; larger values merge more. 0.5 was
            hand-tuned for 2D T-c diagrams — lower it (e.g. 0.2) when disconnected
            stable segments within one phase are incorrectly merged.
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
    bdf = df.query("border")
    # go from a table of mu/c/T points that are on the phase boundaries to a table where the two points that are at the same mu/T are grouped together
    # use this information to add 'transition' column; handles also the case where border points are at mu=+-inf, there we have only one point
    tdf = bdf.groupby(["mu", "T"])[["c", "phase"]].apply(reduce, include_groups=False)
    # immediately explode again to go back to our familiar representation, but now with the added 'transition' column
    tdf = tdf.reset_index().explode(["c", "phase"]).infer_objects().reset_index(drop=True)

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
        tdf["transition_unit"] = []
        tdf["border_segment"] = []

    return tdf
