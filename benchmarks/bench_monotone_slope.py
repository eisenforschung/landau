"""Cost and guarantee of ``SoftplusSurface2DInterpolator(monotone_slope=True)``.

The free slope ``b_i(T)`` is a free polynomial in 1/T and can reverse direction
(sharpen, then soften again as it cools) -- unphysical for a convex well.
``monotone_slope=True`` reparametrises it through the same non-negative softplus
link the amplitude already uses,

    b_i(T) = s_i * [ softplus(B_i0) + sum_{k>=1} softplus(B_ik) * w^k ],
    w = max((1/T) - min_train(1/T), 0),

so ``|b_i(T)|`` is non-increasing in T by construction at every temperature.
Softplus (not sigmoid) is the link: it is unbounded above and keeps the coupled
solve unconstrained, so the default ``lm`` fast path is retained and the slope is
free to grow at the cold end.

This benchmark fits the public class both ways on three representative surfaces and
reports, per surface:

* fit wall-time (best of 3) and its ratio -- the convergence cost of the link;
* worst-over-T RMSE normalised by the well depth -- the accuracy cost;
* ``max_T d|b|/dT`` over a wide sweep incl. extrapolation -- positive for the free
  fit when the well's sharpening is non-monotone, ~0 (the guarantee) for monotone.

Run with ``python benchmarks/bench_monotone_slope.py`` from the repo root
(``[test]`` extras).
"""
from __future__ import annotations

import time

import numpy as np

from landau.interpolate import SoftplusSurface2DInterpolator
from landau.interpolate.softplus import _softplus


# --------------------------------------------------------------------------- #
# representative surfaces (entropy-removed free energy H(T, c))
# --------------------------------------------------------------------------- #
def in_family(T, c, c0=0.36):
    """In-family softplus V, sharpness b(T) = 4 + 30/T (monotone in 1/T)."""
    b = 4.0 + 30.0 * (1000.0 / np.asarray(T, float))
    u = np.asarray(c, float) - c0
    return -0.2 + 0.05 * _softplus(b * u) + 0.03 * _softplus(-b * u)


def sharp_v(T, c, c0=0.36):
    """Asymmetric hard V, branch slopes ~ 1/T (monotone sharpening, worst case)."""
    T, u = np.asarray(T, float), np.asarray(c, float) - c0
    sL, sR = 0.03 + 0.22 * (1000.0 / T), 0.01 + 0.09 * (1000.0 / T)
    return np.where(u < 0, sL * (-u), sR * u)


def humped_b(T, c, c0=0.36):
    """In-family V whose sharpness is *humped* in 1/T -- a deliberately
    non-monotone truth, where the constraint genuinely binds."""
    w = 1000.0 / np.asarray(T, float)
    b = np.maximum(6.0 + 25.0 * w - 6.0 * w ** 2, 1.0)
    u = np.asarray(c, float) - c0
    return -0.2 + 0.05 * _softplus(b * u) + 0.03 * _softplus(-b * u)


CASES = {"in_family": in_family, "sharp_v": sharp_v, "humped_b": humped_b}
NT, NC, TLO, THI, CLO, CHI = 80, 13, 250.0, 1500.0, 0.30, 0.45


def grid():
    Tg, cg = np.linspace(TLO, THI, NT), np.linspace(CLO, CHI, NC)
    return np.repeat(Tg, NC), np.tile(cg, NT), Tg, cg


def nrmse(surface, H, Tg, cg):
    worst = 0.0
    for Tq in Tg:
        truth = H(Tq, cg)
        depth = max(float(truth.max() - truth.min()), 1e-12)
        pred = np.asarray(surface.slice_at(Tq)(cg))
        worst = max(worst, np.sqrt(np.mean((pred - truth) ** 2)) / depth)
    return worst


def max_slope_increase(surface, Tlo, Thi):
    """Largest forward step of ``|b_i(T)|`` as T rises (over all terms), on a wide
    sweep including extrapolation.  > 0 means the sharpness reverses somewhere."""
    Ts = np.linspace(Tlo, Thi, 300)
    babs = np.array([np.abs(surface.slice_at(float(t)).b) for t in Ts])
    return float(np.diff(babs, axis=0).max())


def best_time(fn, reps=3):
    best, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    print(f"grid: {NT} temperatures x {NC} concentrations, T in [{TLO:.0f}, {THI:.0f}]")
    print("slope sweep for d|b|/dT spans [50, 5000] K (heavy extrapolation)\n")
    header = (f"{'surface':11s} {'mode':9s} {'fit [ms]':>9s} {'ratio':>6s} "
              f"{'nrmse':>10s} {'max d|b|/dT':>12s}")
    print(header)
    print("-" * len(header))
    for name, H in CASES.items():
        T, c, Tg, cg = grid()
        f = H(T, c)
        rows = {}
        for mode in (False, True):
            itp = SoftplusSurface2DInterpolator(n_softplus=2, monotone_slope=mode)
            t, surface = best_time(lambda itp=itp: itp.fit(T, c, f))
            rows[mode] = (t, nrmse(surface, H, Tg, cg), max_slope_increase(surface, 50.0, 5000.0))
        t_free = rows[False][0]
        for mode, label in ((False, "free"), (True, "monotone")):
            t, e, mono = rows[mode]
            ratio = t / t_free
            print(f"{name:11s} {label:9s} {t*1e3:9.0f} {ratio:5.2f}x "
                  f"{e:10.2e} {mono:+12.2e}")
        print()


if __name__ == "__main__":
    main()
