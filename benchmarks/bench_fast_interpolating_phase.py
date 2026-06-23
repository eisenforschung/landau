"""Wall-time and accuracy of ``FastInterpolatingPhase`` vs ``SlowInterpolatingPhase``.

Both solve the same problem -- ``phi(dmu) = min_c [ fe(c) - T*S(c) - dmu*c ]``
over a phase's concentration range. ``SlowInterpolatingPhase`` runs one
``scipy.optimize.brute`` (20-point grid + Nelder-Mead polish) per scalar ``dmu``;
``FastInterpolatingPhase`` vectorises over the whole ``dmu`` array with a shared
free-energy curve per ``T`` and a logit-space Newton polish.

The driver mirrors how ``calc_phase_diagram`` calls a phase: a scalar-``T`` loop,
each over a ``dmu`` array, querying ``semigrand_potential`` then ``concentration``.

Run with ``python benchmarks/bench_fast_interpolating_phase.py`` from the repo
root. Requires the ``[test]`` extras.
"""
from __future__ import annotations

import time

import numpy as np

from landau.phases import (
    LinePhase,
    SlowInterpolatingPhase,
    FastInterpolatingPhase,
    S,
)


# ---------- representative phases ----------

def case_polyfit_interior():
    """Interior phase, range (0.3, 0.7) -> PolyFit interpolation (sigma-like)."""
    pts = [
        LinePhase("p0", 0.30, -2.30),
        LinePhase("p1", 0.40, -2.50),
        LinePhase("p2", 0.50, -2.60),
        LinePhase("p3", 0.60, -2.55),
        LinePhase("p4", 0.70, -2.40),
    ]
    return pts


def case_redlich_kister():
    """Full-range phase with terminals -> RedlichKister interpolation."""
    pts = [
        LinePhase("A", 0.00, 0.00),
        LinePhase("q1", 0.25, -0.20),
        LinePhase("mid", 0.50, -0.30),
        LinePhase("q3", 0.75, -0.18),
        LinePhase("B", 1.00, 0.00),
    ]
    return pts


def case_double_well():
    """A landscape with two competing minima (miscibility gap)."""
    pts = [
        LinePhase("A", 0.00, 0.00),
        LinePhase("q1", 0.25, 0.05),
        LinePhase("mid", 0.50, 0.02),
        LinePhase("q3", 0.75, 0.05),
        LinePhase("B", 1.00, 0.00),
    ]
    return pts


CASES = {
    "polyfit_interior": case_polyfit_interior,
    "redlich_kister": case_redlich_kister,
    "double_well": case_double_well,
}

TS = np.linspace(300.0, 1600.0, 14)
MU = np.linspace(-3.0, 3.0, 120)


def _clear_caches(cls):
    # the lru_caches are class attributes shared across equal frozen instances
    for attr in ("_find_phi_c_scalar", "_find_phi_c_cached", "_get_interpolation"):
        f = cls.__dict__.get(attr)
        if f is not None and hasattr(f, "cache_clear"):
            f.cache_clear()


def _drive(phase):
    phi, c = [], []
    for T in TS:
        phi.append(np.asarray(phase.semigrand_potential(float(T), MU)))
        c.append(np.asarray(phase.concentration(float(T), MU)))
    return np.array(phi), np.array(c)


def _time(make_phase, reps=3):
    best = np.inf
    out = None
    for _ in range(reps):
        phase = make_phase()
        for base in type(phase).__mro__:
            _clear_caches(base)
        t0 = time.perf_counter()
        out = _drive(phase)
        best = min(best, time.perf_counter() - t0)
    return best, out


def _gold(phase, T):
    """True minimum on a dense grid, for accuracy reference."""
    fe = phase._get_interpolation(float(T))
    a, b = phase.concentration_range
    cs = np.linspace(a, b, 60001)
    val = fe(cs)[:, None] - T * S(cs)[:, None] - MU[None, :] * cs[:, None]
    i = val.argmin(axis=0)
    return val[i, np.arange(len(MU))]


def main():
    print(f"driver: {len(TS)} temperatures x {len(MU)} dmu  (semigrand + concentration)\n")
    header = f"{'case':18s} {'slow [ms]':>10s} {'fast [ms]':>10s} {'speedup':>8s} " \
             f"{'slow err':>10s} {'fast err':>10s}"
    print(header)
    print("-" * len(header))
    for name, factory in CASES.items():
        pts = factory()
        t_slow, (ps, _) = _time(lambda: SlowInterpolatingPhase(name=name, phases=factory()))
        t_fast, (pf, _) = _time(lambda: FastInterpolatingPhase(name=name, phases=factory()))

        ref = SlowInterpolatingPhase(name=name, phases=pts)
        gold = np.array([_gold(ref, T) for T in TS])
        slow_err = np.abs(ps - gold).max()
        fast_err = np.abs(pf - gold).max()

        print(f"{name:18s} {t_slow*1e3:10.1f} {t_fast*1e3:10.1f} "
              f"{t_slow / t_fast:7.0f}x {slow_err:10.2e} {fast_err:10.2e}")


if __name__ == "__main__":
    main()
