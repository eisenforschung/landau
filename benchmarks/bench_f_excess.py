"""Cost of the ``f_excess`` reference in ``calc_phase_diagram``.

``_f_excess`` evaluates every phase at each distinct temperature of the frame,
far past the sampled ``mu`` window, to find the pure-component free energies.
The refiners add rows at off-grid temperatures, so a refined 2-D frame holds
many more distinct temperatures than the grid.  This times ``_f_excess`` alone
against the whole ``calc_phase_diagram`` call on two systems:

* ``ideal``: three ``IdealSolution`` phases (closed form, cheap per call);
* ``fast``: three ``FastInterpolatingPhase`` phases over Redlich-Kister fits,
  which rebuild their concentration grid on every scalar-``T`` call, so they
  are the expensive case.

Run with ``python benchmarks/bench_f_excess.py`` from the repo root.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from landau.calculate import _f_excess, calc_phase_diagram
from landau.interpolate import RedlichKister
from landau.phases import FastInterpolatingPhase, IdealSolution, LinePhase, kB


def ideal_phases():
    fcc = IdealSolution("fcc", LinePhase("fccA", 0, -3.00, 1.0 * kB), LinePhase("fccB", 1, -2.00, 1.1 * kB))
    hcp = IdealSolution("hcp", LinePhase("hcpA", 0, -2.975, 1.8 * kB), LinePhase("hcpB", 1, -1.95, 1.1 * kB))
    liq = IdealSolution("liquid", LinePhase("liquidA", 0, -2.75, 5.0 * kB), LinePhase("liquidB", 1, -1.75, 4.4 * kB))
    return [hcp, fcc, liq]


def fast_phases():
    def phase(name, eA, sA, eB, sB, mix, smix):
        lines = [LinePhase(f"{name}A", 0, eA, sA * kB),
                 LinePhase(f"{name}M", 0.5, (eA + eB) / 2 + mix, (sA + sB) / 2 * kB + smix * kB),
                 LinePhase(f"{name}B", 1, eB, sB * kB)]
        return FastInterpolatingPhase(name, lines, interpolator=RedlichKister(1))
    return [phase("fcc", -3.00, 1.0, -2.00, 1.1, -0.05, 0.0),
            phase("hcp", -2.975, 1.8, -1.95, 1.1, -0.02, 0.0),
            phase("liquid", -2.75, 5.0, -1.75, 4.4, -0.08, 0.2)]


def bench(name, phases, repeat=3):
    Ts = np.linspace(200, 1000, 25)
    mus = np.linspace(0.5, 1.5, 60)
    total = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        df = calc_phase_diagram(phases, Ts, mus, keep_unstable=True)
        total.append(time.perf_counter() - t0)
    by_name = {p.name: p for p in phases}
    part = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _f_excess(df, by_name)
        part.append(time.perf_counter() - t0)
    n_T = df.query("-inf<mu<inf")["T"].nunique()
    print(f"{name:6s} rows={len(df):6d} distinct T={n_T:5d}  "
          f"calc_phase_diagram {min(total):6.2f} s  _f_excess {min(part):6.3f} s ({min(part) / min(total):.0%})")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    bench("ideal", ideal_phases())
    bench("fast", fast_phases())
