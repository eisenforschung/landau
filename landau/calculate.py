"""
Calculates phase diagrams from sets of Phases.
"""

from functools import partial
import numbers
import warnings

import numpy as np
import pandas as pd
import scipy.optimize as so
from scipy.spatial import Delaunay
from typing import Iterable

from scipy.constants import Boltzmann, eV

kB = Boltzmann / eV

from .phases import Phase, AbstractLinePhase


def find_one_point(phase1, phase2, potential, var_range):
    """
    Find a exact phase transition between to phases.

    Args:
        phase1, phase2 (:class:`landau.phase.Phase`):
            the two phases
        potential (callable):
            function that given a phase and an intensive state variable returns a thermodynamic potential
        var_range (tuple of float):
            interval to search for the transition
    """
    return so.root_scalar(
        lambda x: potential(phase1, x) - potential(phase2, x), bracket=var_range, x0=np.mean(var_range), xtol=1e-6
    ).root


def find_mu_one_point(phase1, phase2, mu_range, T):
    """
    Extract chemical potential of a single phase equilibrium.
    """
    mu = find_one_point(phase1, phase2, lambda p, mu: p.semigrand_potential(T, mu), mu_range)
    return [
        {"mu": mu, "phi": phase1.semigrand_potential(T, mu), "c": phase1.concentration(T, mu), "phase": phase1.name},
        {"mu": mu, "phi": phase2.semigrand_potential(T, mu), "c": phase2.concentration(T, mu), "phase": phase2.name},
    ]


def find_T_one_point(phase1, phase2, T_range, mu):
    T = find_one_point(phase1, phase2, lambda p, T: p.semigrand_potential(T, mu), T_range)
    return [
        {"T": T, "phi": phase1.semigrand_potential(T, mu), "c": phase1.concentration(T, mu), "phase": phase1.name},
        {"T": T, "phi": phase2.semigrand_potential(T, mu), "c": phase2.concentration(T, mu), "phase": phase2.name},
    ]


def find_all_points(stable_df, phases, by="mu"):
    """
    Map find_one_point over all estimated equilibria at a given T or mu.
    """
    assert by in ["T", "mu"], "Wrong by value"
    stable_df = stable_df.sort_values(by).reset_index(drop=True)
    boundary_guesses = stable_df.index[(stable_df.phase != stable_df.phase.shift(-1).ffill())]
    if by == "mu":
        boundaries = [
            # {"mu": -np.inf, "c": 0, "phase": stable_df.phase.iloc[0]},
            # {"mu": np.inf, "c": 1, "phase": stable_df.phase.iloc[-1]},
        ]
    else:
        boundaries = [
            # {"T": stable_df["T"].min(), "c": stable_df.c.iloc[0], "phase": stable_df.phase.iloc[0]},
            # {"T": stable_df["T"].max(), "c": stable_df.c.iloc[-1], "phase": stable_df.phase.iloc[-1]},
        ]
    for g in boundary_guesses:
        r1 = stable_df.loc[g]
        r2 = stable_df.loc[g + 1]
        p1 = phases[r1.phase]
        p2 = phases[r2.phase]
        match by:
            case "mu":
                mus = sorted([r1.mu, r2.mu])
                refinements = find_mu_one_point(
                    p1,
                    p2,
                    mus,
                    r1["T"],
                )
            case "T":
                Ts = sorted([r1["T"], r2["T"]])
                refinements = find_T_one_point(
                    p1,
                    p2,
                    Ts,
                    r1["mu"],
                )
        # the find*point functions find the point where the potentials of the two phases are equal, but a third phase
        # could be lower in potential, so only add the refined points if they are truly stable
        # FIXME: technically this only needs to be done once, since the refinements share T/mu
        for phase in phases.values():
            if phase.name in (r1.phase, r2.phase):
                continue
            for point in refinements:
                T = point.get("T", r1["T"])
                mu = point.get("mu", r1["mu"])
                if phase.semigrand_potential(T, mu) < point["phi"]:
                    break
            else:
                continue
            break
        else:
            boundaries.extend(refinements)
    df = pd.DataFrame(boundaries)
    if len(df) > 0:
        df = df.sort_values(by)
    return df


def find_triangle(phases, cand):
    """
    Assumes two of the three points in cand are of the same phase.

    Args:
        phases (Phases): all phases
        cand (dataframe): subset of phase diagram dataframe of the three points of the triangle
    """
    # one of p1, p2 will be the peak of the triangle, the other one the center of the base
    p1, p2 = cand.groupby("phase")[["T", "mu"]].mean().to_numpy()

    def project(t):
        T, mu = p1 + (p2 - p1) * t
        return T, mu

    phase1, phase2 = [phases[p] for p in cand.phase.unique()]
    try:
        t = find_one_point(phase1, phase2, lambda phase, t: phase.semigrand_potential(*project(t)), (0, 1))
    except ValueError:
        warnings.warn(f"Failed to refine triangle between {p1} and {p2} of phases {cand.phase.unique()}!", stacklevel=2)
        return []
    T, mu = p1 + (p2 - p1) * t
    if T < 0:
        return []
    phi = phase1.semigrand_potential(T, mu)
    # check that no other phase is lower than the refined boundary
    if any(p.semigrand_potential(T, mu) < phi for p in phases.values() if p.name not in (phase1.name, phase2.name)):
        # TODO: could try and refine here on the boundary (p1, more_stable_phase) and (p2, more_stable_phase)
        return []
    return [
        {"T": T, "mu": mu, "phi": phases[p].semigrand_potential(T, mu), "c": phases[p].concentration(T, mu), "phase": p}
        for p in cand.phase.unique()
    ]


def refine_phase_diagram(df, phases, min_c=0, max_c=1):
    """Add additional points to a coarse phase diagram by searching for exact transitions."""
    udf = df.query("not stable").reset_index(drop=True)
    udf["border"] = False
    df = df.query("stable").reset_index(drop=True)
    df["border"] = False
    df["refined"] = "no"
    data = [df, udf]
    multiple_mus = len(df["mu"].unique()) > 1
    multiple_ts = len(df["T"].unique()) > 1
    if multiple_mus and multiple_ts:
        # declare edges of the sampling window as borders so to not confuse get_transitions, debatably hacky
        df.loc[df["T"] == df["T"].min(), "border"] = True
        df.loc[df["T"] == df["T"].max(), "border"] = True
        # left and right edges as well, set here to +-inf to make sure the
        # cluster algo below separates top and left and right edges, even
        # hackier
        left = df.loc[df["mu"] == df["mu"].min()][["phase", "T"]]
        left["mu"] = -np.inf
        left["c"] = min_c
        left["border"] = True
        left["stable"] = True
        data.append(left)
        right = df.loc[df["mu"] == df["mu"].max()][["phase", "T"]]
        right["mu"] = +np.inf
        right["c"] = max_c
        right["border"] = True
        right["stable"] = True
        data.append(right)
        # Main idea:
        # - tessellate input points
        # - count the number of unique phases in each triangle
        # - if > 1 there must be at least one phase transition in the triangle
        # - find_triangle assumes (erroneously) that it can be found on the vector connecting the peak of the triangle
        # to the center of the base
        # - if = 3 there's probably a triple point in there (but doesn't need to be actually, should check that)
        dela = Delaunay(df[["mu", "T"]])
        coex = df.phase.to_numpy()[dela.simplices]
        phase_counts = np.array([len(set(x)) for x in coex])
        line_candidates = [df.iloc[i] for i in dela.simplices[phase_counts == 2]]
        trip_candidates = [df.iloc[i] for i in dela.simplices[phase_counts == 3]]
        # you'd think this to be faster, but somehow un/pickling the phases is very slow, likely because it has to do it
        # for each triangle and involves fitting the phases all over
        # with ProcessPoolExecutor(4) as pool:
        #     ddf = pool.map(partial(find_triangle, phases), line_candidates)
        ddf = map(partial(find_triangle, phases), line_candidates)
        ddf = pd.DataFrame(sum(ddf, []))
        ddf["stable"] = True
        ddf["border"] = True
        ddf["refined"] = "delaunay"
        data.append(ddf)

        def refine_triples(tr):
            T, mu = tr[["T", "mu"]].mean()
            p1, p2, p3 = (phases[p] for p in tr.phase.unique())

            def triplemin(x):
                T, mu = x
                phi1 = p1.semigrand_potential(T, mu)
                phi2 = p2.semigrand_potential(T, mu)
                phi3 = p3.semigrand_potential(T, mu)
                return abs(phi1 - phi2) + abs(phi2 - phi3) + abs(phi3 - phi1)

            T, mu = so.fmin(triplemin, (T, mu), disp=False)
            if T < 0:
                return []
            return [
                {
                    "T": T,
                    "mu": mu,
                    "phi": phases[p].semigrand_potential(T, mu),
                    "c": phases[p].concentration(T, mu),
                    "phase": p,
                }
                for p in tr.phase.unique()
            ]

        tdf = []
        for tri in trip_candidates:
            tdf.extend(refine_triples(tri))
        tdf = pd.DataFrame(tdf)
        tdf["stable"] = True
        tdf["border"] = True
        tdf["refined"] = "delaunay-triple"
        data.append(tdf)
    else:
        if multiple_mus:
            mdf = (
                df.groupby("T", group_keys=True)
                .apply(find_all_points, phases=phases, by="mu", include_groups=True)
                .reset_index()
            )
            mdf["stable"] = True
            mdf["border"] = True
            mdf["refined"] = "mu"
            data.append(mdf.drop("level_1", axis="columns"))
        if multiple_ts:
            Tdf = (
                df.groupby("mu", group_keys=True)
                .apply(find_all_points, phases=phases, by="T", include_groups=True)
                .reset_index()
            )
            Tdf["stable"] = True
            Tdf["border"] = True
            Tdf["refined"] = "T"
            data.append(Tdf.drop("level_1", axis="columns"))
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

    resi = so.minimize(lambda x: +c(x[0]), x0=[0], tol=tolerance, method="BFGS")
    resa = so.minimize(lambda x: -c(x[0]), x0=[0], tol=tolerance, method="BFGS")
    mu0 = resi.x[0]
    mu1 = resa.x[0]
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
    return si.interp1d(cc, mm)(np.linspace(c0, c1, samples)), c0, c1


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


from sklearn.cluster import AgglomerativeClustering


def reduce(dd):
    dd = dd.sort_values("c")
    return pd.Series(
        {
            "transition": "-".join(dd.phase.tolist()),
            "c": dd.c.tolist(),
            "phase": dd.phase.tolist(),
        }
    )


def cluster(dd, eps=0.01, use_mu=True):
    t = dd["T"]
    # Guard against isothermal segments
    if t.min() != t.max():
        t = (t - t.min()) / (t.max() - t.min())
    ids = pd.Series(np.zeros_like(dd.index), index=dd.index)
    # picking eps is a pain and HDBSCAN not available in my env yet
    # cluster = DBSCAN(eps=eps, min_samples=3)
    cluster = AgglomerativeClustering(
        n_clusters=None,
        # FIXME: hand optimized value; smaller values tend to partition the
        # same transition too often
        distance_threshold=0.5,
        linkage="single",
    )
    if use_mu:
        # on the left and right side of the phase diagram refining adds points with
        # mu +- inf, which chokes the cluster methods, but we know they should be
        # their own segments, so special case them below
        F = np.isfinite(dd.mu)
        if F.any() and sum(F) >= 2:
            ids.loc[F] = cluster.fit_predict(np.transpose([t.loc[F], dd.c.loc[F], dd.mu.loc[F]]))
        m = ids.max()
        ids.loc[dd.mu == +np.inf] = m + 1
        ids.loc[dd.mu == -np.inf] = m + 2
    else:
        ids.loc[:] = cluster.fit_predict(np.transpose([t, dd.c]))
    return ids


def get_transitions(df):
    """
    Identify "continuous" two-phase transition lines in mu/T space, i.e. transitions between the same two phases and along which mu/T are continuous.

    Useful for plotting below, but potentially also to augment the existing refining routines and
    acquire additional Free energies from calphy/etc. to improve the diagram.
    """
    bdf = df.query("border")
    # go from a table of mu/c/T points that are on the phase boundaries to a table where the two points that are at the same mu/T are grouped together
    # use this information to add 'transition' column; handles also the case where border points are at mu=+-inf, there we have only one point
    tdf = bdf.groupby(["mu", "T"])[["c", "phase"]].apply(reduce)
    # immediately explode again to go back to our familiar representation, but now with the added 'transition' column
    tdf = tdf.reset_index().explode(["c", "phase"]).infer_objects().reset_index(drop=True)

    # cluster points that are assigned as one transition, because the same transition can appear multiple times in "disconnected" manner in a phase
    # diagram, e.g. a solid solution in contact with the melt interrupted by a higher melting intermetallic
    tdf["transition_unit"] = tdf.groupby("transition", group_keys=False).apply(cluster, include_groups=False)
    tdf["border_segment"] = tdf[["transition", "transition_unit"]].apply(
        lambda r: "_".join(map(str, r.tolist())), axis="columns"
    )

    return tdf
