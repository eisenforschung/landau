"""Accuracy of ``SoftplusSurface2DInterpolator`` on V-shaped free-energy wells.

The 2-D softplus fit used to get stuck: on wells whose sharpness varies strongly
with temperature it placed the knee/branch slopes badly, worst at the temperature
extremes.  Two things drive the fix and this benchmark measures both in isolation:

1. **Slope basis.**  A convex well sharpens as it cools, so the softplus slope
   ``b`` scales with 1/T.  Over a wide T-range a low-order polynomial in T cannot
   track that, while one in 1/T can.  ``_fit(..., b_basis="T" | "1/T")`` runs the
   *same* algorithm with only the slope basis swapped, isolating the effect.
2. **Seeding.**  Each coupled fit is seeded from a convex-well-aware per-slice fit
   (knee at the data minimum, opposite-slope softplus pair) at the coldest and
   central slices, polished with ``x_scale='jac'`` -- so the sharp-knee landscape
   does not trap it.  This is what the public class does.

Ground truth is a convex V whose two branches each have their own slope and
curvature, all varying with T (``VSurface``), and an exactly in-family softplus V
whose sharpness follows 1/T (``SoftplusVSurface``).  Error is the per-slice RMSE
normalised by the well depth, reported as the mean and 95th percentile over all
temperatures, plus the worst knee-position error.

Run with ``python benchmarks/bench_softplus_surface_minima.py`` from the repo
root (``[test]`` extras).  The final block also checks the public
``SoftplusSurface2DInterpolator`` reproduces the ``b_basis='1/T'`` standalone.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from landau.interpolate import SoftplusSurface2DInterpolator
from landau.interpolate.softplus import _fit_slice, _softplus, _softplus_inv


# --------------------------------------------------------------------------- #
# ground-truth V-shaped wells
# --------------------------------------------------------------------------- #
def _lin1T(T, a, b):
    return a + b * (1000.0 / T)


class VSurface:
    """Asymmetric, curved V; each branch's slope and curvature vary with T."""

    def __init__(self, name, clo, chi, c0, sL, sR, qL, qR, c0_drift=0.0):
        self.name, self.clo, self.chi, self.c0_base = name, clo, chi, c0
        self.sL, self.sR, self.qL, self.qR, self.c0_drift = sL, sR, qL, qR, c0_drift

    def c0(self, T):
        return self.c0_base + self.c0_drift * (1000.0 / T - 1000.0 / 600.0)

    def H(self, T, c):
        T, c = np.asarray(T, float), np.asarray(c, float)
        u = c - self.c0(T)
        left = self.sL(T) * (-u) + self.qL(T) * u ** 2
        right = self.sR(T) * u + self.qR(T) * u ** 2
        return np.where(u < 0.0, left, right)

    def depth(self, T):
        return float(np.max([self.H(T, self.clo), self.H(T, self.chi)]))


class SoftplusVSurface:
    """Exactly in-family softplus V whose sharpness b(T) = b0 + b1/T."""

    def __init__(self, name, clo, chi, c0, b0, b1, aL, aR):
        self.name, self.clo, self.chi, self.c0_base = name, clo, chi, c0
        self.b0, self.b1, self.aL, self.aR = b0, b1, aL, aR

    def c0(self, T):
        return self.c0_base

    def H(self, T, c):
        T, c = np.asarray(T, float), np.asarray(c, float)
        b = self.b0 + self.b1 * (1000.0 / T)
        u = c - self.c0_base
        return self.aR * np.logaddexp(0.0, b * u) + self.aL * np.logaddexp(0.0, -b * u)

    def depth(self, T):
        edges = np.max([self.H(T, self.clo), self.H(T, self.chi)])
        return float(edges - self.H(T, self.c0_base))


def sample(ds, nT, nc, Tlo, Thi, noise=0.0, seed=0):
    Tg, cg = np.linspace(Tlo, Thi, nT), np.linspace(ds.clo, ds.chi, nc)
    T, c = np.repeat(Tg, nc), np.tile(cg, nT)
    H = ds.H(T, c)
    if noise:
        rng = np.random.default_rng(seed)
        H = H + rng.normal(scale=noise * np.array([ds.depth(t) for t in Tg]).repeat(nc))
    return T, c, H, Tg, cg


# --------------------------------------------------------------------------- #
# standalone surface fit with a switchable slope basis (T vs 1/T)
# --------------------------------------------------------------------------- #
def _fit(T, c, H, n=2, a_order=1, b_order=2, c_order=1, o_order=2, b_basis="1/T"):
    Tm, Ts = T.mean(), T.std() or 1.0
    cm, cs = c.mean(), c.std() or 1.0
    s = (1.0 / T) if b_basis == "1/T" else T
    sm, ss = s.mean(), s.std() or 1.0
    Tn, Sn, cn = (T - Tm) / Ts, (s - sm) / ss, (c - cm) / cs
    na, nb, nc, no = a_order + 1, b_order + 1, c_order + 1, o_order + 1
    per = na + nb + nc
    VTa = np.vander(Tn, na, increasing=True)
    VSb = np.vander(Sn, nb, increasing=True)
    VTc = np.vander(Tn, nc, increasing=True)
    VTo = np.vander(Tn, no, increasing=True)

    def model(p):
        out = VTo @ p[n * per:]
        for i in range(n):
            base = i * per
            a = _softplus(VTa @ p[base:base + na])
            b = VSb @ p[base + na:base + na + nb]
            cc = VTc @ p[base + na + nb:base + per]
            out = out + a * _softplus(b * (cn + cc))
        return out

    uT = np.unique(T)
    best = None
    for Tseed in (uT[0], uT[len(uT) // 2]):
        m = np.isclose(T, Tseed)
        a0, b0, c0, off0 = _fit_slice(cn[m], H[m], n, 4000)
        p0 = np.zeros(n * per + no)
        for i in range(n):
            p0[i * per] = _softplus_inv(a0[i])
            p0[i * per + na] = b0[i]
            p0[i * per + na + nb] = c0[i]
        p0[n * per] = off0
        res = least_squares(lambda p: model(p) - H, p0, x_scale="jac", max_nfev=4000)
        if best is None or res.cost < best.cost:
            best = res

    def slice_at(Tq, cq):
        Tn_ = (Tq - Tm) / Ts
        Sn_ = (((1.0 / Tq) if b_basis == "1/T" else Tq) - sm) / ss
        cn_ = (np.asarray(cq, float) - cm) / cs
        pv = np.polynomial.polynomial.polyval
        out = np.full(cn_.shape, pv(Tn_, best.x[n * per:]))
        for i in range(n):
            base = i * per
            a = _softplus(pv(Tn_, best.x[base:base + na]))
            b = pv(Sn_, best.x[base + na:base + na + nb])
            cc = pv(Tn_, best.x[base + na + nb:base + per])
            out = out + a * _softplus(b * (cn_ + cc))
        return out

    return slice_at


def metrics(ds, slice_fn, Tg, cg):
    nr, ke = [], []
    cd = np.linspace(cg.min(), cg.max(), 401)
    for T in Tg:
        depth = max(ds.depth(T), 1e-12)
        nr.append(np.sqrt(np.mean((slice_fn(T, cg) - ds.H(T, cg)) ** 2)) / depth)
        ke.append(abs(cd[np.argmin(slice_fn(T, cd))] - ds.c0(T)))
    return np.mean(nr), np.percentile(nr, 95), np.max(ke)


SURFACES = [
    VSurface("sharp_asym", 0.26, 0.40, 0.33,
             lambda T: _lin1T(T, 0.03, 0.22), lambda T: _lin1T(T, 0.01, 0.09),
             lambda T: _lin1T(T, 0.0, 0.05), lambda T: _lin1T(T, 0.0, 0.25), c0_drift=0.01),
    VSurface("wide_asym", 0.30, 0.45, 0.36,
             lambda T: _lin1T(T, 0.0, 0.45), lambda T: _lin1T(T, 0.0, 0.18),
             lambda T: _lin1T(T, 0.0, 0.10), lambda T: _lin1T(T, 0.0, 0.6), c0_drift=0.015),
    SoftplusVSurface("infamily_1overT", 0.30, 0.45, 0.36, b0=4.0, b1=40.0, aL=0.05, aR=0.03),
]


def main():
    print(f"{'surface':16s} {'Trange':>11s} | "
          f"{'b~poly(T)  m/p95/kerr':>24s} | {'b~poly(1/T) m/p95/kerr':>24s}")
    for ds in SURFACES:
        Tlo, Thi = (150.0, 1500.0) if "wide" in ds.name or "infamily" in ds.name else (300.0, 900.0)
        T, c, H, Tg, cg = sample(ds, nT=120, nc=13, Tlo=Tlo, Thi=Thi, noise=0.03)
        mT = metrics(ds, _fit(T, c, H, b_basis="T"), Tg, cg)
        m1 = metrics(ds, _fit(T, c, H, b_basis="1/T"), Tg, cg)
        print(f"{ds.name:16s} {Tlo:5.0f}-{Thi:<5.0f} | "
              f"{mT[0]:7.3f} {mT[1]:7.3f} {mT[2]:6.4f} | {m1[0]:7.3f} {m1[1]:7.3f} {m1[2]:6.4f}")

    # the public class uses the 1/T basis -- confirm it matches the standalone
    ds = SURFACES[-1]
    T, c, H, Tg, cg = sample(ds, nT=120, nc=13, Tlo=150.0, Thi=1500.0, noise=0.0)
    pub = SoftplusSurface2DInterpolator(n_softplus=2).fit(T, c, H)
    m = metrics(ds, lambda Tq, cq: np.asarray(pub.slice_at(Tq)(cq)), Tg, cg)
    print(f"\npublic SoftplusSurface2DInterpolator on {ds.name}: "
          f"nRMSE mean={m[0]:.4f} p95={m[1]:.4f} (recovers in-family b~1/T surface)")


if __name__ == "__main__":
    main()
