"""cProfile the refiners on the testplots example systems.

Drives ``calc_phase_diagram(..., refine=True)`` on the same phase systems the
visual-review plots use (``tests/integration/testplots.py``) and prints, per
system, the wall-clock and the top cumulative-time frames. Use it to see where
``refine_phase_diagram`` spends its time and to check a refiner change.

Run ``python benchmarks/profile_refiners.py`` (all systems) or
``python benchmarks/profile_refiners.py basics_mu`` for one.

Systems:
* ``basics_mu`` — hcp/fcc/liquid ideal solutions on a 100x100 (mu, T) grid;
  cheap ``semigrand_potential``, so refiner bookkeeping (the Delaunay
  tessellation shared across refiners) dominates.
* ``toy`` — regular-solution ``FastInterpolatingPhase`` liquid + intermediate
  solid; expensive per-T solves, so the Clausius-Clapeyron trace's isothermal
  root-finding dominates.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

import numpy as np

import landau.phases as ldp
import landau.calculate as ldc
import landau.interpolate as ldi


def basics_mu():
    """hcp / fcc / liquid ideal solutions, 100x100 grid (from Basics.ipynb)."""
    fcc = ldp.IdealSolution("fcc", ldp.LinePhase("fccA", 0, -3.00, 1.0 * ldp.kB),
                            ldp.LinePhase("fccB", 1, -2.00, 1.1 * ldp.kB))
    hcp = ldp.IdealSolution("hcp", ldp.LinePhase("hcpA", 0, -2.975, 1.8 * ldp.kB),
                            ldp.LinePhase("hcpB", 1, -1.95, 1.1 * ldp.kB))
    lqd = ldp.IdealSolution("liquid", ldp.LinePhase("lA", 0, -2.75, 5.0 * ldp.kB),
                            ldp.LinePhase("lB", 1, -1.75, 4.4 * ldp.kB))
    Ts = np.linspace(200, 1000, 100)
    mus = np.linspace(0.5, 1.5, 100)
    return lambda: ldc.calc_phase_diagram([hcp, fcc, lqd], Ts, mu=mus)


def toy():
    """Regular-solution liquid + intermediate solid (from Toy.ipynb)."""
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
    c = np.linspace(0, 1, 75)[1:-1]
    mu = 1 + ldp.kB * 4000 * np.log(c / (1 - c))
    return lambda: ldc.calc_phase_diagram([rliq, sol, s3], np.linspace(500, 1000, 40), mu, refine=True)


SYSTEMS = {"basics_mu": basics_mu, "toy": toy}


def run(name: str, builder, repeats: int = 3, top: int = 15) -> None:
    fn = builder()
    best = np.inf
    df = None
    for _ in range(repeats):
        t = time.perf_counter()
        df = fn()
        best = min(best, time.perf_counter() - t)
    print(f"\n=== {name}: {best * 1e3:.0f} ms  ({len(df)} rows) ===")
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(top)
    # keep only landau frames plus the header for a compact view
    for line in s.getvalue().splitlines():
        if "landau/" in line or "ncalls" in line or "function calls" in line:
            print(line)


def main() -> None:
    names = sys.argv[1:] or list(SYSTEMS)
    for name in names:
        run(name, SYSTEMS[name])


if __name__ == "__main__":
    main()
