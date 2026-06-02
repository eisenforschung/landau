"""
Calculates phase diagrams from sets of Phases.
"""

import numbers
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann, eV
from sklearn.cluster import AgglomerativeClustering


from .phases import Phase, AbstractLinePhase
from .refine import Refiner, default_refiners, _find_one_point as find_one_point  # noqa: F401


kB = Boltzmann / eV


__all__ = ["calc_phase_diagram", "get_transitions", "cluster_T_c", "cluster_T_c_mu"]


_PHASE_UNIT_SEP = "_"


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


def _split_stable(df):
    udf = df.query("not stable").reset_index(drop=True)
    udf["border"] = False
    sdf = df.query("stable").reset_index(drop=True)
    sdf["border"] = False
    sdf["refined"] = "no"
    return sdf, udf


def _border_edges(df, min_c, max_c):
    """Mark T extremes as border and emit synthetic +-inf mu edges (2D only)."""
    df.loc[df["T"] == df["T"].min(), "border"] = True
    df.loc[df["T"] == df["T"].max(), "border"] = True
    left = df.loc[df["mu"] == df["mu"].min()][["phase", "T"]].copy()
    left["mu"] = -np.inf
    left["c"] = min_c
    left["border"] = True
    left["stable"] = True
    right = df.loc[df["mu"] == df["mu"].max()][["phase", "T"]].copy()
    right["mu"] = +np.inf
    right["c"] = max_c
    right["border"] = True
    right["stable"] = True
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
    import numpy as np
    # semigrand canonical "average" concentration
    # use this to avoid discontinuities and be phase agnostic

    def c(mu):
        phis = np.array([p.semigrand_potential(T, mu) for p in phases])
        conc = np.array([p.concentration(T, mu) for p in phases])
        phis -= phis.min(axis=0)
        beta = 1 / (kB * T)
        prob = np.exp(-beta * phis)
        prob /= prob.sum(axis=0)
        ci = (prob * conc).sum(axis=0)
        return ci

    mu0 = so.brute(lambda x: +c(x[0]), ranges=[(-10, 10)], Ns=200)[0]
    mu1 = so.brute(lambda x: -c(x[0]), ranges=[(-10, 10)], Ns=200)[0]
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
        dataframe of phase points
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
    if refine:
        min_c = pdf.c.min()
        max_c = pdf.c.max()
        pdf = refine_phase_diagram(pdf, phases, min_c=min_c, max_c=max_c)
    pdf["f"] = pdf.phi + pdf.mu * pdf.c

    def sub(dd):
        dd = dd.query("-inf<mu<inf")
        c0 = dd.c.min()
        c1 = dd.c.max()
        f0 = dd.query("c==@c0").f.min()
        f1 = dd.query("c==@c1").f.min()
        return dd.f - (f0 * (1 - dd.c) + f1 * dd.c)

    fex = pdf.groupby("T", group_keys=False).apply(sub, include_groups=False)
    if len(Ts) > 1:
        pdf["f_excess"] = fex
    else:
        # thank you pandas, this saved me -10min of my life.
        pdf["f_excess"] = fex.T
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
        res = tdf.groupby("transition", group_keys=False).apply(
            lambda g: cluster_T_c_mu(g, distance_threshold=0.5),  # 0.5 hand-tuned
            include_groups=False,
        )
        if isinstance(res, pd.DataFrame):
             # sometimes pandas returns a DataFrame instead of a Series when only one group exists
             res = res.stack().reset_index(level=0, drop=True)
        tdf["transition_unit"] = res
        tdf["border_segment"] = _join_phase_unit(tdf["transition"], tdf["transition_unit"])
    else:
        tdf["transition_unit"] = []
        tdf["border_segment"] = []

    return tdf
