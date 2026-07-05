"""Reproduce the Au-Cu phase diagram of

    B. Sundman, S. G. Fries, W. A. Oates,
    "A thermodynamic assessment of the Au-Cu system",
    Calphad 22 (1998) 335-354.

The paper gives two assessments.  **Assessment I** (Table 1, eqs 6-9) models the
fcc-based phases with the four-sublattice Compound Energy Formalism, capturing the
ordered L1_0 (AuCu) and L1_2 (Au3Cu, AuCu3) superstructures inside the disordered
A1 matrix (Figs. 1, 3).  **Assessment II** (eq 10) drops the ordering and models
the liquid and the disordered fcc as plain substitutional solutions -- a better fit
to the liquidus/solidus but no order/disorder region.  Both share the same liquid,
a substitutional solution (eqs 1-3).

This script builds both assessments from the paper's parameters as code -- no
database file is read -- and draws them side by side through
``calc_phase_diagram`` / ``plot_phase_diagram`` for comparison.

Units: the four-sublattice compound energies and interaction parameters are per
mole of formula (4 sites) in J/mol, so the per-formula ``G`` is divided by NS=4
sites; the liquid and Assessment-II fcc are per mole of atoms (1 site).  Everything
is converted to eV/atom by dividing by 96485.33 J/mol.

Run:  python benchmarks/reproduce_sundman_aucu.py
"""

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from landau import CompoundEnergyPhase
from landau.calculate import calc_phase_diagram
from landau.plot import plot_phase_diagram
from landau.refine import DelaunayLineRefiner

JMOL = 96485.332  # J/mol per eV/atom
NS = 4            # sites per fcc formula unit

# --- pure-element liquid lattice stabilities relative to fcc (SGTE, Dinsdale 1991) ---
# per mole of atoms, J/mol; zero at the element melting points (Au 1337.33 K, Cu 1357.77 K)


def _G_liq_Au(T):
    return (12552.45 - 9.385866 * T) / JMOL


def _G_liq_Cu(T):
    return (12964.735 - 9.511904 * T - 5.83932e-21 * T**7) / JMOL


def _redlich_kister(y, T, coeffs):
    """Substitutional Redlich-Kister excess (eqs 2-3), per atom (eV).  ``y`` is x_Cu."""
    y = np.asarray(y, dtype=float)
    x = y[..., 0]
    dd = (1 - x) - x  # x_Au - x_Cu
    e = 0.0
    for v, L in enumerate(coeffs):
        e = e + L(T) * dd**v
    return (1 - x) * x * e / JMOL


def _liquid():
    """The liquid, a substitutional solution shared by both assessments (eq 6)."""
    coeffs = (lambda T: -27900 - T, lambda T: 4730.0, lambda T: 3500 + 3.5 * T)
    return CompoundEnergyPhase(
        name="liquid",
        site_multiplicities=(1.0,),
        endmember_energies={(0,): _G_liq_Au, (1,): _G_liq_Cu},
        excess=lambda y, T: _redlich_kister(y, T, coeffs),
    )


# --- Assessment I: four-sublattice CEF fcc (Table 1, eqs 7-9) -----------------
_u1, _u2, _u3, _u4, _u5, _u6 = -7590, -9590, -9900, 10.32, 10.62, -1.565


def _G_Au3Cu(T):
    return (3 * _u1 + 3 * _u4 * T + 3 * _u6 * T * np.log(T)) / NS / JMOL


def _G_Au2Cu2(T):
    return (4 * _u2 + 4 * _u5 * T + 4 * _u6 * T * np.log(T)) / NS / JMOL


def _G_AuCu3(T):
    return (3 * _u3 + 3 * _u4 * T + 3 * _u6 * T * np.log(T)) / NS / JMOL


_BY_NCU = {0: lambda T: 0.0, 1: _G_Au3Cu, 2: _G_Au2Cu2, 3: _G_AuCu3, 4: lambda T: 0.0}
_ENDMEMBERS = {tuple(int(b) for b in cfg): _BY_NCU[sum(cfg)] for cfg in np.ndindex(2, 2, 2, 2)}


def _excess_cef(y, T):
    """Regular (eq 8) + reciprocal (eq 9) interactions, per atom (eV); broadcasts over y."""
    y = np.asarray(y, dtype=float)
    e_reg = (3940 + 10.32 * T) * np.sum(y * (1 - y), axis=-1)
    e_rec = 0.0
    for r, s in combinations(range(4), 2):
        t, u = (i for i in range(4) if i not in (r, s))
        e_rec = e_rec + (1 - y[..., r]) * y[..., r] * (1 - y[..., s]) * y[..., s] * (-18 * T - 15900 * y[..., t] * y[..., u])
    return (e_reg + e_rec) / NS / JMOL


_CEF = dict(site_multiplicities=(0.25, 0.25, 0.25, 0.25), endmember_energies=_ENDMEMBERS, excess=_excess_cef)


def assessment_i_phases():
    """Liquid + four competing fcc phases: disordered A1 and the ordered superstructures."""
    return [
        _liquid(),
        CompoundEnergyPhase(name="fcc", orderings=(), **_CEF),
        CompoundEnergyPhase(name="Au3Cu", orderings=((1, 0, 0, 0),), include_disordered_seed=False, **_CEF),
        CompoundEnergyPhase(name="AuCu", orderings=((1, 1, 0, 0),), include_disordered_seed=False, **_CEF),
        CompoundEnergyPhase(name="AuCu3", orderings=((1, 1, 1, 0),), include_disordered_seed=False, **_CEF),
    ]


# --- Assessment II: substitutional liquid + disordered fcc (eq 10) ------------


def assessment_ii_phases():
    """Liquid + a single disordered fcc, both plain substitutional solutions (no ordering)."""
    fcc_coeffs = (lambda T: -28000 + 78.8 * T - 10 * T * np.log(T), lambda T: 6000.0)
    fcc = CompoundEnergyPhase(
        name="fcc",
        site_multiplicities=(1.0,),
        endmember_energies={(0,): lambda T: 0.0, (1,): lambda T: 0.0},
        excess=lambda y, T: _redlich_kister(y, T, fcc_coeffs),
    )
    return [_liquid(), fcc]


def _draw(ax, phases, title):
    temperatures = np.arange(300, 1420, 30.0)
    mu = np.linspace(-0.7, 0.7, 61)
    df = calc_phase_diagram(phases, temperatures, mu=mu, refine=[DelaunayLineRefiner()])
    plot_phase_diagram(df, poly_method="segments", legend=True, inline_legend=False, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(300, 1420)
    ax.set_xlabel("Mole fraction Cu")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    return df.query("stable").phase.value_counts().to_dict()


def main():
    print(f"AuCu (L1_0) end-member energy at 300 K: {_G_Au2Cu2(300) * JMOL:8.1f} J/mol-atom  (Fig 5 ~ -9000)")
    print(f"Au melting point (Gliq = Gfcc): {12552.45 / 9.385866:7.1f} K  (paper 1337.33)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    c1 = _draw(ax1, assessment_i_phases(), "Assessment I (CEF: order/disorder)")
    c2 = _draw(ax2, assessment_ii_phases(), "Assessment II (substitutional, no order)")
    fig.suptitle("Au-Cu (Sundman, Fries & Oates 1998)")
    fig.tight_layout()
    out = "benchmarks/_aucu_phase_diagram.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"  Assessment I  stable points: {c1}")
    print(f"  Assessment II stable points: {c2}")


if __name__ == "__main__":
    main()
