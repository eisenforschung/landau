"""Offset of the sampled-tangent ``f_excess`` reference from the pure components.

The reference ``calc_phase_diagram`` used before ``_terminal_free_energies``
took, per temperature, the lowest tangent ``phi`` (c = 0) / ``phi + mu``
(c = 1) at the extreme sample of every phase whose sampled c-range came
within 1% of the sampled span.  It is reimplemented here on the unrefined
grid and compared against the pure-component free energies on the
three-phase ideal system of #499, for a wide and a narrow ``mu`` window.

Run with ``python benchmarks/f_excess_reference_offset.py`` from the repo root.
"""

from __future__ import annotations

import numpy as np

from landau.calculate import _terminal_free_energies, calc_phase_diagram
from landau.phases import IdealSolution, LinePhase, kB


def sampled_tangent_references(grid):
    """Old per-T reference: lowest end tangent among phases within 1% of the sampled span."""
    out = {}
    for T, dd in grid.query("-inf<mu<inf").groupby("T"):
        span = dd.c.max() - dd.c.min()
        lo, hi = dd.c.min() + 0.01 * span, dd.c.max() - 0.01 * span
        f0 = f1 = np.inf
        for _, g in dd.groupby("phase"):
            if g.c.min() <= lo:
                i = g.c.idxmin()
                f0 = min(f0, dd.at[i, "phi"])
            if g.c.max() >= hi:
                i = g.c.idxmax()
                f1 = min(f1, dd.at[i, "phi"] + dd.at[i, "mu"])
        out[T] = (f0, f1)
    return out


def main():
    fcc = IdealSolution("fcc", LinePhase("fccA", 0, -3.00, 1.0 * kB), LinePhase("fccB", 1, -2.00, 1.1 * kB))
    hcp = IdealSolution("hcp", LinePhase("hcpA", 0, -2.975, 1.8 * kB), LinePhase("hcpB", 1, -1.95, 1.1 * kB))
    liq = IdealSolution("liquid", LinePhase("liquidA", 0, -2.75, 5.0 * kB),
                        LinePhase("liquidB", 1, -1.75, 4.4 * kB))
    phases = [hcp, fcc, liq]
    Ts = np.linspace(200, 1000, 25)
    for lo, hi in ((0.5, 1.5), (0.7, 1.3)):
        mus = np.linspace(lo, hi, 60)
        grid = calc_phase_diagram(phases, Ts, mus, refine=False, keep_unstable=True)
        old = sampled_tangent_references(grid)
        new = _terminal_free_energies(phases, Ts, mus.min(), mus.max())
        d = np.array([old[T] for T in Ts]) - new
        print(f"mu {lo}..{hi}: old - f(0) in [{d[:, 0].min():+.1e}, {d[:, 0].max():+.1e}]"
              f"   old - f(1) in [{d[:, 1].min():+.1e}, {d[:, 1].max():+.1e}]"
              f"   worst at T = {Ts[np.abs(d).max(axis=1).argmax()]:.0f} K")


if __name__ == "__main__":
    main()
