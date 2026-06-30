"""Sampling density of ``ClausiusClapeyronRefiner`` on a flat-in-mu boundary.

The predictor-corrector step size ``_dT_adapt`` bounds the *mu* drift per
step (``half_width / |dmu/dT|``, clamped to ``[dT_min, dT_max]``). That keeps
the corrector bracket valid but ties the T-sampling density to the
coexistence slope: a boundary that is nearly flat in mu (equal phase
entropies, so ``dmu/dT -> 0`` by Clausius-Clapeyron ``dmu/dT = -ds/dc``)
always saturates at ``dT_max`` and is sampled coarsely -- even where its
concentration sweeps a lot, which is what actually gets plotted in the c-T
diagram. This is the asymmetry behind a densely-sampled solid/liquid line
next to a coarsely-sampled intermetallic.

``dc_max`` adds a concentration-drift floor: the step also shrinks so the
plotted ``c`` moves at most ``dc_max`` per step. A straight constant-c
boundary keeps ``dc/dT ~ 0`` and the cap never engages, so nothing is
over-sampled.

This driver isolates the regime with a toy phase whose composition drifts
linearly with T at fixed mu, so the boundary is exactly flat in mu while c
sweeps. Run with ``python benchmarks/cc_sampling_density.py`` from the repo
root; pass ``--plot out.png`` to also write the c-T scatter.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from landau.phases import Phase
from landau.refine import ClausiusClapeyronRefiner, _InterCandidate


@dataclass(frozen=True)
class DriftLinePhase(Phase):
    """``phi(T, mu) = e - mu * c(T)`` with ``c(T) = c0 + slope*(T - T0)``.

    Then ``c = -dphi/dmu = c(T)`` exactly: a line-like phase whose plotted
    composition drifts with T at fixed mu.
    """

    e: float = 0.0
    c0: float = 0.5
    slope: float = 0.0
    T0: float = 300.0

    def _c(self, T):
        return self.c0 + self.slope * (np.asarray(T, float) - self.T0)

    def semigrand_potential(self, T, mu):
        return self.e - np.asarray(mu, float) * self._c(T)

    def concentration(self, T, mu):
        return self._c(T) + 0.0 * np.asarray(mu, float)


def trace(dc_max: float):
    # Equal e -> coexistence at mu = 0 for every T (flat boundary); P's
    # composition sweeps 0.8 -> 0.1 over the sampled T range.
    P = DriftLinePhase(name="P", e=-2.0, c0=0.8, slope=-0.7 / 1100.0)
    Q = DriftLinePhase(name="Q", e=-2.0, c0=0.05, slope=0.0)
    cand = _InterCandidate(
        phase1="P", phase2="Q", T_seed=850.0,
        mu_bracket=(-0.05, 0.05), T_bracket=(800.0, 900.0),
        T_min=300.0, T_max=1400.0,
        proj_p1=(800.0, -0.1), proj_p2=(900.0, 0.1),
    )
    pts = ClausiusClapeyronRefiner(dc_max=dc_max).solve(cand, {"P": P, "Q": Q})
    pts = sorted(pts, key=lambda p: p.T)
    T = np.array([p.T for p in pts])
    c = np.array([float(P.concentration(p.T, p.mu)) for p in pts])
    mu = np.array([p.mu for p in pts])
    return T, c, mu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None, help="write a c-T scatter PNG here")
    args = ap.parse_args()

    runs = [("mu-drift only (dc_max=inf)", 1e9), ("dc_max=0.02", 0.02)]
    series = []
    for label, dc_max in runs:
        T, c, mu = trace(dc_max)
        max_dc = float(np.abs(np.diff(c)).max()) if len(c) > 1 else 0.0
        print(f"{label:>26}: n={len(T):3d}  max|dc/step|={max_dc:.4f}  "
              f"|mu|max={np.abs(mu).max():.2e}")
        series.append((label, T, c))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        for (label, T, c), color in zip(series, ["tab:red", "tab:green"]):
            ax.plot(c, T, "o-", ms=4, color=color, label=f"{label} (n={len(T)})")
        ax.set_xlabel("c"); ax.set_ylabel("T")
        ax.set_title("CC sampling density on a flat-mu, c-sweeping boundary")
        ax.legend()
        fig.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
