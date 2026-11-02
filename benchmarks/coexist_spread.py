"""Own-phase potential spread of every refined point, the quantity
``_COEXIST_TOL`` bounds in ``landau.refine._dominated``.

Coexisting phases share one semigrand potential, so a refined point's own
phases should agree in ``phi`` to within the refiners' convergence. This driver
wraps ``_dominated`` to record, for every point it is asked about, the spread
``max(phi) - min(phi)`` over the point's own phases and whether a phase outside
them is lower, then renders every diagram in ``tests/integration/testplots.py``
and prints the distribution split into two-phase points, triples, and points
naming one phase twice.

Run ``python benchmarks/coexist_spread.py`` from the repo root. The rendered
PNGs go to a temporary directory.
"""
from __future__ import annotations

import os
import runpy
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np

import landau.refine as refine

_records = []
_dominated = refine._dominated


def _recording_dominated(pt, phases):
    own = pt.phase_names()
    phi = [phases[n].semigrand_potential(pt.T, pt.mu) for n in own]
    if np.all(np.isfinite(phi)):
        hi = max(phi)
        outside = any(
            p.semigrand_potential(pt.T, pt.mu) < hi for p in phases.values() if p.name not in own
        )
        _records.append((hi - min(phi), len(own), len(getattr(pt, "phases", own)), outside))
    return _dominated(pt, phases)


def main():
    refine._dominated = _recording_dominated
    with tempfile.TemporaryDirectory() as out:
        sys.argv = ["testplots.py", "--out", out, "--poly-method", "concave"]
        runpy.run_path(os.path.join("tests", "integration", "testplots.py"), run_name="__main__")
    refine._dominated = _dominated

    spread, n_set, n_tuple, outside = (np.array(c) for c in zip(*_records))
    print(f"\n_COEXIST_TOL = {refine._COEXIST_TOL:.0e} eV; {len(spread)} refined points")
    groups = {
        "two-phase": (n_set == 2) & (n_tuple == 2),
        "triple": n_set == 3,
        "one phase twice": (n_set == 2) & (n_tuple == 3),
    }
    for label, sel in groups.items():
        for dominated in (False, True):
            s = spread[sel & (outside == dominated)]
            if len(s):
                print(
                    f"{label:16s} {'outside-dominated' if dominated else 'kept by outside test':21s} "
                    f"N={len(s):6d} median={np.median(s):.1e} max={s.max():.1e}"
                )


if __name__ == "__main__":
    main()
