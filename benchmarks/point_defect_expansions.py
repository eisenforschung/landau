"""Diagnostic comparing a host phase against its exact and low-temperature
point-defect expansions across temperatures.

Builds the B2 intermetallic of ``notebooks/PointDefects.ipynb`` (a stoichiometric
``LinePhase`` host with antisite + vacancy defects on two sublattices) three
ways:

* ``host``  — the bare ``LinePhase`` (no defects),
* ``exact`` — ``PointDefectedPhase`` with ``PointDefectSublattice`` (full
  site competition, ``-kB*T*ln(1 + sum_i z_i)``),
* ``LTE``   — ``PointDefectedPhase`` with ``LowTemperatureExpansionSublattice``
  (dilute leading term, ``-kB*T*sum_i z_i``).

For a range of temperatures it plots the concentration c(dmu) and the defect
contribution to the semi-grand potential phi(dmu) - phi_host(dmu) for all three,
and reports the half-window of dmu over which the exact and LTE concentrations
agree to 0.01. The expansion tracks the exact model while defects are dilute
(low T / small |dmu|) and departs once site competition matters. The raw,
unstitched LTE site fractions (dotted) overshoot c = 1 and send phi to -inf;
``PointDefectedPhase`` clamps c to [0, 1] and continues phi as the tangent line
past each saturation point (dashed), which both bounds phi and keeps
c = -dphi/dmu consistent. The agreement window is wide at low T and shrinks as
T rises and defects are no longer dilute even at stoichiometry.

Run: ``python benchmarks/point_defect_expansions.py [--out DIR]``
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, eV

from landau.phases import LinePhase
from landau.phases.pointdefects import (
    ConstantPointDefect,
    LowTemperatureExpansionSublattice,
    PointDefectSublattice,
    PointDefectedPhase,
)

kB = Boltzmann / eV

TEMPERATURES = (300.0, 800.0, 1500.0, 3000.0)
AGREE_TOL = 0.01


def build_phases():
    """Host LinePhase plus its exact and LTE point-defect expansions."""
    host = LinePhase("AB", fixed_concentration=0.5, line_energy=-0.40, line_entropy=1.0 * kB)

    # B2: alpha (A sites) carries B antisites + vacancies, beta (B sites) the counterpart.
    # The alpha defects are given lower formation energies so the model is asymmetric.
    alpha = [
        ConstantPointDefect("B_alpha", excess_energy=0.28, excess_entropy=0.0, excess_solutes=+1),
        ConstantPointDefect("V_alpha", excess_energy=0.45, excess_entropy=0.0, excess_solutes=0),
    ]
    beta = [
        ConstantPointDefect("A_beta", excess_energy=0.30, excess_entropy=0.0, excess_solutes=-1),
        ConstantPointDefect("V_beta", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
    ]

    def defected(sublattice_cls):
        # the same defect objects feed either expansion; only the combiner differs
        return PointDefectedPhase(
            name=sublattice_cls.__name__,
            line_phase=host,
            sublattices=[
                sublattice_cls(name="alpha", sublattice=0, sublattice_fraction=0.5, defects=alpha),
                sublattice_cls(name="beta", sublattice=1, sublattice_fraction=0.5, defects=beta),
            ],
        )

    return host, defected(PointDefectSublattice), defected(LowTemperatureExpansionSublattice)


def agreement_halfwindow(dmu, dc):
    """Largest |dmu| around 0 over which |c_exact - c_lte| stays below AGREE_TOL."""
    inside = np.abs(dc) < AGREE_TOL
    # walk outward from dmu = 0 until the tolerance is first broken
    zero = int(np.argmin(np.abs(dmu)))
    lo = zero
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    hi = zero
    while hi < len(dmu) - 1 and inside[hi + 1]:
        hi += 1
    return min(abs(dmu[lo]), abs(dmu[hi]))


def main(out_dir):
    host, exact, lte = build_phases()
    dmu = np.linspace(-0.6, 0.6, 600)

    fig, axes = plt.subplots(2, len(TEMPERATURES), figsize=(4 * len(TEMPERATURES), 7), sharex=True)

    print(f"{'T [K]':>8} {'agree |dmu| [eV]':>18}")
    for j, T in enumerate(TEMPERATURES):
        c_host = np.full_like(dmu, host.concentration(T, dmu))
        c_exact = exact.concentration(T, dmu)
        c_lte = lte.concentration(T, dmu)  # clamped to [0, 1]
        phi_host = host.semigrand_potential(T, dmu)
        dphi_exact = exact.semigrand_potential(T, dmu) - phi_host
        dphi_lte = lte.semigrand_potential(T, dmu) - phi_host  # phi tangent-extended

        # raw, unstitched LTE (no clamp, no tangent continuation) to show the fix
        c_lte_raw = host.line_concentration + sum(
            s.concentration_contribution(T, dmu) for s in lte.sublattices
        )
        dphi_lte_raw = sum(s.semigrand_potential_contribution(T, dmu) for s in lte.sublattices)

        print(f"{T:>8.0f} {agreement_halfwindow(dmu, c_exact - c_lte):>18.3f}")

        ax_c, ax_phi = axes[0, j], axes[1, j]

        ax_c.axhline(1.0, color="0.8", lw=0.8, zorder=0)
        ax_c.axhline(0.0, color="0.8", lw=0.8, zorder=0)
        ax_c.plot(dmu, c_host, color="0.6", lw=1.2, label="host")
        ax_c.plot(dmu, c_exact, color="C0", lw=1.6, label="exact")
        ax_c.plot(dmu, c_lte_raw, color="C3", lw=1.0, ls=":", alpha=0.7, label="LTE (raw)")
        ax_c.plot(dmu, c_lte, color="C3", lw=1.6, ls="--", label="LTE")
        ax_c.set_title(f"T = {T:.0f} K")
        ax_c.set_ylim(-0.05, 1.6)

        ax_phi.axhline(0.0, color="0.6", lw=1.2, label="host")
        ax_phi.plot(dmu, dphi_exact, color="C0", lw=1.6, label="exact")
        ax_phi.plot(dmu, dphi_lte_raw, color="C3", lw=1.0, ls=":", alpha=0.7, label="LTE (raw)")
        ax_phi.plot(dmu, dphi_lte, color="C3", lw=1.6, ls="--", label="LTE")
        ax_phi.set_xlabel(r"$\Delta\mu$ (eV)")
        ax_phi.set_ylim(-0.5, 0.03)

        if j == 0:
            ax_c.set_ylabel("concentration $c$")
            ax_phi.set_ylabel(r"$\phi - \phi_{\mathrm{host}}$ (eV/atom)")
            ax_c.legend(loc="upper left", fontsize=9)

    fig.suptitle("Host vs exact vs low-temperature-expansion point-defect model")
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "point_defect_expansions.png"
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".", help="directory to write the PNG into")
    args = parser.parse_args()
    main(args.out)
