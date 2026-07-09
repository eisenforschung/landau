"""Where the ``ClausiusClapeyronRefiner`` trace spends its time.

This driver backs the timing analysis on the vectorized coarse-then-refine CC
trace (the ``_bisect_vectorized`` densification). It is a *diagnostic* record,
not a win: on the phases currently in the library the vectorization does not
speed the refiner up, and the numbers here say why. It becomes a win only for a
phase whose ``semigrand_potential`` genuinely batches over an *array of
distinct T* with non-trivial per-element cost.

Four measurements, run from the repo root with
``python benchmarks/bench_cc_vectorized.py``. The old-vs-new end-to-end table
in the PR description is measurement 0 run once on ``main`` and once on this
branch (same machine).

0. **end-to-end.** Best-of-5 ``refine_phase_diagram`` with only the CC refiner,
   on a cheap (``IdealSolution``) and an expensive (``FastInterpolatingPhase``)
   two-phase system, with the emitted point count.

1. **propose vs. trace (cheap phase).** On an ``IdealSolution`` solid/liquid
   grid, how the wall-clock splits between ``propose`` and the trace. Since
   #335 (numpy-backed ``_Simplex``) ``propose`` is a few percent and the
   trace's ``semigrand_potential`` calls dominate.

2. **eval attribution (expensive phase).** On a ``FastInterpolatingPhase`` pair
   the expensive ``_solve_fixed_T`` calls are counted by refiner call site. They
   land almost entirely in the *scalar* coarse walk / seed root-finding
   (``<lambda>`` inside ``_refine_step``), not in the vectorized densification.

3. **densification batching (vectorizing phase).** For ``IdealSolution`` —
   which broadcasts ``semigrand_potential`` over ``T`` — the densification's
   ``_bisect_vectorized`` calls carry many elements each, i.e. the batching the
   design is built to exploit does happen; it just does not dominate runtime.
"""
from __future__ import annotations

import time
from collections import Counter
import traceback

import numpy as np
from scipy.constants import Boltzmann, eV

from landau.phases import LinePhase, IdealSolution, FastInterpolatingPhase
from landau.interpolate import RedlichKister
from landau.calculate import calc_phase_diagram, refine_phase_diagram
from landau.refine import ClausiusClapeyronRefiner, _delaunay_simplices

kB = Boltzmann / eV

Ts = np.linspace(200, 1800, 25)
mus = np.linspace(-0.3, 0.3, 21)


def _best(fn, repeats=5):
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def ideal_system():
    solid = IdealSolution(
        "solid", LinePhase("A", 0, -2.0, 1.0 * kB), LinePhase("B", 1, -3.0, 1.5 * kB))
    liquid = IdealSolution(
        "liquid", LinePhase("A(l)", 0, -1.9, 2.5 * kB), LinePhase("B(l)", 1, -2.9, 2.2 * kB))
    return [solid, liquid], {"solid": solid, "liquid": liquid}


def fast_system():
    def mk(name, e, s):
        cs = [0, 0.25, 0.5, 0.75, 1.0]
        lps = [LinePhase(f"{name}{i}", c, e + 0.2 * c * (1 - c), s) for i, c in enumerate(cs)]
        return FastInterpolatingPhase(
            name=name, phases=lps, add_entropy=True, interpolator=RedlichKister(2))
    a, b = mk("alpha", -2.0, 1.0 * kB), mk("beta", -1.9, 2.6 * kB)
    return [a, b], {"alpha": a, "beta": b}


def end_to_end():
    print("[0] end-to-end refine_phase_diagram, CC refiner only (best of 5)")
    for label, system in [("IdealSolution", ideal_system),
                          ("FastInterpolatingPhase", fast_system)]:
        phases, mapping = system()
        df = calc_phase_diagram(phases, Ts=Ts, mu=mus, refine=False, keep_unstable=False)
        out = refine_phase_diagram(df, mapping, refiners=[ClausiusClapeyronRefiner()])
        cc = out[out["refined"] == "clausius-clapeyron"]
        t = _best(lambda: refine_phase_diagram(df, mapping, refiners=[ClausiusClapeyronRefiner()]))
        print(f"    {label:22s} {t * 1e3:7.1f} ms   "
              f"output points: {cc.groupby(['T', 'mu']).ngroups}")


def propose_vs_trace():
    phases, mapping = ideal_system()
    df = calc_phase_diagram(phases, Ts=Ts, mu=mus, refine=False, keep_unstable=False)
    r = ClausiusClapeyronRefiner()
    n_simplices = len(list(_delaunay_simplices(df)))
    t_propose = _best(lambda: list(r.propose(df)))
    t_run = _best(lambda: r.run(df, mapping))
    print("[1] propose vs. trace (IdealSolution, cheap phi)")
    print(f"    grid={len(df)} points, {n_simplices} simplices")
    print(f"    propose(): {t_propose * 1e3:6.1f} ms   "
          f"full run(): {t_run * 1e3:6.1f} ms   "
          f"propose share: {t_propose / t_run:4.0%}")


def eval_attribution():
    phases, mapping = fast_system()
    df = calc_phase_diagram(phases, Ts=Ts, mu=mus, refine=False, keep_unstable=False)
    site = Counter()
    orig = FastInterpolatingPhase._solve_fixed_T

    def counted(self, T, dmu):
        last = "?"
        for fr in traceback.extract_stack():
            if "refine.py" in fr.filename:
                last = fr.name
        site[last] += 1
        return orig(self, T, dmu)

    FastInterpolatingPhase._solve_fixed_T = counted
    try:
        out = refine_phase_diagram(df, mapping, refiners=[ClausiusClapeyronRefiner()])
    finally:
        FastInterpolatingPhase._solve_fixed_T = orig
    cc = out[out["refined"] == "clausius-clapeyron"]
    print("[2] expensive _solve_fixed_T evals by call site "
          "(FastInterpolatingPhase)")
    print(f"    output points={cc.groupby(['T', 'mu']).ngroups}   "
          f"total evals={sum(site.values())}")
    for name, n in site.most_common():
        print(f"      {name:20} {n}")


def densification_batching():
    phases, mapping = ideal_system()
    df = calc_phase_diagram(phases, Ts=Ts, mu=mus, refine=False, keep_unstable=False)
    sizes = []
    orig = IdealSolution.semigrand_potential

    def counted(self, T, dmu):
        sizes.append(int(np.size(np.broadcast_arrays(np.asarray(T), np.asarray(dmu))[0])))
        return orig(self, T, dmu)

    IdealSolution.semigrand_potential = counted
    try:
        refine_phase_diagram(df, mapping, refiners=[ClausiusClapeyronRefiner()])
    finally:
        IdealSolution.semigrand_potential = orig
    arr = np.array(sizes)
    print("[3] semigrand_potential call sizes (IdealSolution, vectorizes over T)")
    print(f"    calls={arr.size}   scalar(1-elem)={int((arr == 1).sum())}   "
          f"batched(>1)={int((arr > 1).sum())}   max batch={arr.max()}")


def main() -> None:
    end_to_end()
    print()
    propose_vs_trace()
    print()
    eval_attribution()
    print()
    densification_batching()


if __name__ == "__main__":
    main()
