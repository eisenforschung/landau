"""Delaunay2DRefiner vs. the three separate default refiners.

The 2-D default used to run DelaunayTripleRefiner, ClausiusClapeyronRefiner and
MiscibilityGapRefiner separately, each tessellating the (mu, T) grid in its own
propose. Delaunay2DRefiner merges them: one tessellation, one propose that
yields typed candidates, one run that buckets them back per refiner. This
driver times both on the testplots example systems and checks the emitted
transition rows are identical.

Run ``python benchmarks/bench_combined_refiner.py`` from the repo root.
Reference (best of 6, refine_phase_diagram only, on a precomputed coarse frame):

    system        3 separate   combined
    2d_basics_mu      949 ms      498 ms   (~1.9x; tessellation-bound)
    2d_toy           1205 ms     1128 ms   (~6%; per-T-solve-bound)
"""
from __future__ import annotations

import time

import numpy as np

import landau.phases as ldp
import landau.calculate as ldc
import landau.interpolate as ldi
from landau.refine import (
    Delaunay2DRefiner, DelaunayTripleRefiner, ClausiusClapeyronRefiner,
    MiscibilityGapRefiner,
)

SEPARATE = [DelaunayTripleRefiner(), ClausiusClapeyronRefiner(), MiscibilityGapRefiner()]
COMBINED = [Delaunay2DRefiner()]


def basics_mu():
    fcc = ldp.IdealSolution("fcc", ldp.LinePhase("fccA", 0, -3.00, 1.0 * ldp.kB),
                            ldp.LinePhase("fccB", 1, -2.00, 1.1 * ldp.kB))
    hcp = ldp.IdealSolution("hcp", ldp.LinePhase("hcpA", 0, -2.975, 1.8 * ldp.kB),
                            ldp.LinePhase("hcpB", 1, -1.95, 1.1 * ldp.kB))
    lqd = ldp.IdealSolution("liquid", ldp.LinePhase("lA", 0, -2.75, 5.0 * ldp.kB),
                            ldp.LinePhase("lB", 1, -1.75, 4.4 * ldp.kB))
    phases = [hcp, fcc, lqd]
    coarse = ldc.calc_phase_diagram(phases, np.linspace(200, 1000, 100),
                                    mu=np.linspace(0.5, 1.5, 100),
                                    refine=False, keep_unstable=True)
    return coarse, {p.name: p for p in phases}


def toy():
    def line(name, c, fs, ip):
        return ldp.TemperatureDependentLinePhase(
            name, fixed_concentration=c, temperatures=[1, 750, 1000],
            free_energies=fs, interpolator=ip)
    rliq = ldp.FastInterpolatingPhase("liquid", [
        line("l0", 0, [2.00, 1.80, 1.00], ldi.PolyFit(3)),
        line("l2", 0.5, [2.45, 2.00, 1.42], ldi.PolyFit(3)),
        line("l1", 1, [3.00, 2.80, 2.00], ldi.PolyFit(3))])
    sol = ldp.IdealSolution("solid", line("s0", 0, [1.9, 1.6, 1.2], ldi.SGTE(2)),
                            line("s1", 1, [2.9, 2.6, 2.2], ldi.SGTE(2)))
    s3 = line("s3", 0.4, np.array([2.4, 1.85, 1.45]) - 0.05, ldi.SGTE(3))
    phases = [rliq, sol, s3]
    c = np.linspace(0, 1, 75)[1:-1]
    mu = 1 + ldp.kB * 4000 * np.log(c / (1 - c))
    coarse = ldc.calc_phase_diagram(phases, np.linspace(500, 1000, 40), mu,
                                    refine=False, keep_unstable=True)
    return coarse, {p.name: p for p in phases}


def _rows_key(df):
    d = df[df["refined"].isin(["delaunay-triple", "clausius-clapeyron", "miscibility-gap"])]
    return sorted((r, round(t, 9), round(m, 9), p, round(c, 9))
                  for r, t, m, p, c in zip(d["refined"], d["T"], d["mu"], d["phase"], d["c"]))


def _best(coarse, mapping, refiners, repeats=6):
    best = np.inf
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = ldc.refine_phase_diagram(coarse, mapping, refiners=refiners)
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out


def main() -> None:
    print(f"{'system':>14} {'3 separate':>12} {'combined':>10}  identical")
    for name, build in (("2d_basics_mu", basics_mu), ("2d_toy", toy)):
        coarse, mapping = build()
        t_sep, sep = _best(coarse, mapping, SEPARATE)
        t_comb, comb = _best(coarse, mapping, COMBINED)
        ok = _rows_key(sep) == _rows_key(comb)
        print(f"{name:>14} {t_sep:>10.0f}ms {t_comb:>8.0f}ms  {ok}")


if __name__ == "__main__":
    main()
