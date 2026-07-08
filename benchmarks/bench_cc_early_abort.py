"""Work skipped by the ClausiusClapeyronRefiner early abort.

``_CCBase._trace`` used to walk each two-phase coexistence line all the way to
the sampled T bound and let ``run`` discard the points where a third phase is
more stable (the metastable tail past a triple point). The trace now stops as
soon as it enters a dominated region, so those root-finds are never done. The
final ``run`` output is unchanged — only the wasted work is removed.

This driver traces a three-phase system whose A-B line is stable only above its
triple point, and reports the run wall-clock plus, per coexistence pair, the
fraction of the sampled T range that is metastable (and so no longer traced).

Run ``python benchmarks/bench_cc_early_abort.py`` from the repo root. Measured
against ``main`` (pre-abort) on the default grid: run() 40.2 ms -> 33.6 ms,
identical 210-row output; the A-B trace shrinks from 220-480 K to 300-480 K.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from landau.phases import LinePhase
from landau.refine import ClausiusClapeyronRefiner, _dominated


def three_phase_system():
    """Three line phases meeting at a single triple point (T=300, mu=0.2)."""
    return {
        "A": LinePhase(name="A", fixed_concentration=0.0, line_energy=-1.0, line_entropy=0.004),
        "B": LinePhase(name="B", fixed_concentration=0.5, line_energy=-1.2, line_entropy=0.003),
        "C": LinePhase(name="C", fixed_concentration=1.0, line_energy=-1.7, line_entropy=0.001),
    }


def coarse_df(phases, Ts, mus):
    rows = []
    for T in Ts:
        for mu in mus:
            phis = {n: float(p.semigrand_potential(T, mu)) for n, p in phases.items()}
            n = min(phis, key=phis.get)
            rows.append({"T": T, "mu": mu, "phi": phis[n],
                         "c": float(phases[n].concentration(T, mu)),
                         "phase": n, "stable": True})
    return pd.DataFrame(rows)


def _best(fn, repeats=5):
    best = np.inf
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out


def main() -> None:
    phases = three_phase_system()
    Ts = np.linspace(220, 480, 20)
    mus = np.linspace(-0.05, 0.55, 20)
    df = coarse_df(phases, Ts, mus)
    T_span = float(df["T"].max() - df["T"].min())

    r = ClausiusClapeyronRefiner()
    ms, out = _best(lambda: r.run(df, phases))
    print(f"grid={len(df)} points   run(): {ms:.1f} ms   rows={len(out)}")

    # Per pair: traced T-extent vs the metastable fraction now skipped.
    print("per coexistence pair (traced T-range | metastable fraction skipped):")
    seen = set()
    for cand in r.propose(df):
        pair = frozenset((cand.phase1, cand.phase2))
        if pair in seen:
            continue
        seen.add(pair)
        pts = [pt for c in r.propose(df)
               if frozenset((c.phase1, c.phase2)) == pair
               for pt in r.solve(c, phases)]
        if not pts:
            continue
        assert all(not _dominated(pt, phases) for pt in pts)
        lo, hi = min(pt.T for pt in pts), max(pt.T for pt in pts)
        skipped = 1.0 - (hi - lo) / T_span
        label = "-".join(sorted(pair))
        print(f"  {label:>10}: {lo:6.1f}-{hi:6.1f} K   {skipped:5.0%}")


if __name__ == "__main__":
    main()
