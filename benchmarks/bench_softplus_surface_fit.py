"""Wall-time and accuracy of ``SoftplusSurface2DInterpolator.fit``.

Both the current ``fit`` and the ``legacy_fit`` baseline below solve the same
problem -- one coupled nonlinear least squares for

    H(T, c) = off(T) + sum_i a_i(T) softplus( b_i(T) (cn + c_i(T)) )

from the same convex-well-aware per-slice seeds (:func:`_fit_slice`).  They
differ in how much redundant work the coupled solve does:

* ``legacy_fit`` runs *both* seeds to full convergence with the trust-region
  solver and keeps the cheaper -- the worse seed's full solve is wasted work --
  and evaluates the residual and Jacobian in two separate passes (the per-term
  ``softplus`` twice).
* ``fit`` *races* the two seeds (a short polish each, finish only the better),
  caps the per-slice seed polish, and evaluates the residual and Jacobian
  together so the transcendentals are computed once.  On scipy >= 1.16 it also
  drives the coupled solve with ``method='lm'`` (no per-iteration SVD); on older
  scipy it uses the same trust-region solver as ``legacy_fit``.

So on scipy >= 1.16 the speed-up combines less redundant work with the cheaper
solver; on older scipy it is the redundant-work part alone.

The fit is exact for an in-family surface and a close convex approximation for a
sharp V; the reported error is the per-temperature RMSE normalised by the well
depth, worst over all temperatures.

Run with ``python benchmarks/bench_softplus_surface_fit.py`` from the repo root.
Requires the ``[test]`` extras.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from landau.interpolate import SoftplusSurface2DInterpolator
from landau.interpolate.softplus import (
    SoftplusFittedSurface,
    _fit_slice,
    _softplus,
    _sigmoid,
    _standardize,
)


# --------------------------------------------------------------------------- #
# legacy baseline: two full trust-region coupled fits, separate model/jac
# --------------------------------------------------------------------------- #
def legacy_fit(itp: SoftplusSurface2DInterpolator, T, c, f) -> SoftplusFittedSurface:
    """The pre-optimisation coupled fit, for a fair before/after comparison."""
    T = np.asarray(T, float)
    c = np.asarray(c, float)
    f = np.asarray(f, float)
    na, nb, nc, no = itp._orders
    n = itp.n_softplus
    Tn, Tm, Ts = _standardize(T)
    Wn, wm, ws = _standardize(1.0 / T)
    cn, cm, cs = _standardize(c)
    vt = (
        np.vander(Tn, na, increasing=True),
        np.vander(Wn, nb, increasing=True),
        np.vander(Tn, nc, increasing=True),
        np.vander(Tn, no, increasing=True),
    )

    def model(p):
        VTa, VWb, VTc, VTo = vt
        A, B, C, O = itp._unpack(p)
        out = VTo @ O
        for i in range(n):
            out = out + _softplus(VTa @ A[i]) * _softplus((VWb @ B[i]) * (cn + VTc @ C[i]))
        return out

    def jac(p):
        VTa, VWb, VTc, VTo = vt
        A, B, C, O = itp._unpack(p)
        per = na + nb + nc
        J = np.empty((cn.size, n * per + no))
        for i in range(n):
            alpha = VTa @ A[i]
            a = _softplus(alpha)
            sig_a = _sigmoid(alpha)
            b = VWb @ B[i]
            u = cn + VTc @ C[i]
            t = b * u
            sig = _sigmoid(t)
            base = i * per
            J[:, base:base + na] = (_softplus(t) * sig_a)[:, None] * VTa
            J[:, base + na:base + na + nb] = (a * sig * u)[:, None] * VWb
            J[:, base + na + nb:base + per] = (a * sig * b)[:, None] * VTc
        J[:, n * per:] = VTo
        return J

    uT = np.unique(T)
    ridge = 1e-7 * (float(np.abs(f).max()) or 1.0)
    eye = ridge * np.eye(n * (na + nb + nc) + no)
    best = None
    for Tseed in (uT[0], uT[len(uT) // 2]):
        m = np.isclose(T, Tseed)
        p0 = itp._const_init(*_fit_slice(cn[m], f[m], n, itp.max_nfev))
        res = least_squares(
            lambda p: np.concatenate([model(p) - f, ridge * (p - p0)]),
            p0, jac=lambda p: np.vstack([jac(p), eye]),
            x_scale="jac", max_nfev=itp.max_nfev,
        )
        if best is None or res.cost < best.cost:
            best = res
    A, B, C, O = itp._unpack(best.x)
    return SoftplusFittedSurface(A, B, C, O, Tm, Ts, wm, ws, cm, cs)


# --------------------------------------------------------------------------- #
# representative surfaces (entropy-removed free energy H(T, c))
# --------------------------------------------------------------------------- #
def in_family(T, c, c0=0.36):
    """Exactly in-family softplus V whose sharpness b(T) = b0 + b1/T."""
    b = 4.0 + 30.0 * (1000.0 / np.asarray(T, float))
    u = np.asarray(c, float) - c0
    return -0.2 + 0.05 * _softplus(b * u) + 0.03 * _softplus(-b * u)


def sharp_v(T, c, c0=0.36):
    """Asymmetric hard V kink whose branch slopes scale with 1/T (worst case)."""
    T, u = np.asarray(T, float), np.asarray(c, float) - c0
    sL, sR = 0.03 + 0.22 * (1000.0 / T), 0.01 + 0.09 * (1000.0 / T)
    return np.where(u < 0, sL * (-u), sR * u)


def wide_v(T, c, c0=0.36):
    """A wide, mildly curved asymmetric V (each branch curved, drifting knee)."""
    T, u = np.asarray(T, float), np.asarray(c, float) - c0
    sL, sR = 0.45 * (1000.0 / T), 0.18 * (1000.0 / T)
    qL, qR = 0.10 * (1000.0 / T), 0.6 * (1000.0 / T)
    return np.where(u < 0, sL * (-u) + qL * u ** 2, sR * u + qR * u ** 2)


CASES = {
    "in_family": (in_family, 0.30, 0.45),
    "sharp_v": (sharp_v, 0.30, 0.45),
    "wide_v": (wide_v, 0.30, 0.45),
}

NT, NC = 120, 13
TLO, THI = 150.0, 1500.0


def _grid(clo, chi):
    Tg, cg = np.linspace(TLO, THI, NT), np.linspace(clo, chi, NC)
    return np.repeat(Tg, NC), np.tile(cg, NT), Tg, cg


def _nrmse(surf, H, Tg, cg):
    """Worst over T of the per-slice RMSE normalised by the well's depth (its
    peak-to-trough range over the sampled c), so an offset on H does not matter."""
    worst = 0.0
    for Tq in Tg:
        truth = H(Tq, cg)
        depth = max(float(truth.max() - truth.min()), 1e-12)
        pred = np.asarray(surf.slice_at(Tq)(cg))
        worst = max(worst, np.sqrt(np.mean((pred - truth) ** 2)) / depth)
    return worst


def _time(fit_fn, reps=3):
    best = np.inf
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fit_fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    print(f"grid: {NT} temperatures x {NC} concentrations, T in [{TLO:.0f}, {THI:.0f}]\n")
    header = (f"{'surface':12s} {'legacy [ms]':>12s} {'fast [ms]':>10s} {'speedup':>8s} "
              f"{'legacy nrmse':>13s} {'fast nrmse':>11s}")
    print(header)
    print("-" * len(header))
    itp = SoftplusSurface2DInterpolator(n_softplus=2)
    for name, (H, clo, chi) in CASES.items():
        T, c, Tg, cg = _grid(clo, chi)
        f = H(T, c)
        t_leg, leg = _time(lambda: legacy_fit(itp, T, c, f))
        t_fast, fast = _time(lambda: itp.fit(T, c, f))
        e_leg = _nrmse(leg, H, Tg, cg)
        e_fast = _nrmse(fast, H, Tg, cg)
        print(f"{name:12s} {t_leg*1e3:12.0f} {t_fast*1e3:10.0f} {t_leg/t_fast:7.1f}x "
              f"{e_leg:13.2e} {e_fast:11.2e}")


if __name__ == "__main__":
    main()
