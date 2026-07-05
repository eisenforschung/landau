"""Reproduce the Au-Cu order/disorder phase diagram of

    B. Sundman, S. G. Fries, W. A. Oates,
    "A thermodynamic assessment of the Au-Cu system",
    Calphad 22 (1998) 335-354,   Assessment I, Figures 1 and 3.

The fcc-based phases -- disordered A1 and the ordered L1_0 (AuCu) / L1_2 (Au3Cu,
AuCu3) superstructures -- all share one four-sublattice Compound Energy Formalism
energetics; ordering is an internal (site-fraction) degree of freedom.  A single
partitioning ``CompoundEnergyPhase`` (``orderings=None``) would represent the whole
fcc field, with the order/disorder transitions as intra-phase c(mu) jumps.  To draw
the diagram with landau's phase-vs-phase pipeline we instead let each ordering
compete as its own basin-pinned phase (same energetics, seeded from one ordered
corner) -- the global minimum, and hence the phase diagram, is identical (the lower
envelope of the basins) but each phase is a clean region ``calc_phase_diagram`` can
resolve.

All parameters are given here as code -- no database file is read.  Units: the
CALPHAD parameters are per mole of formula (4 sites) in J/mol; ``CompoundEnergyPhase``
works per atom in eV, so the per-formula energies are divided by NS=4 sites and by
96485.33 J/mol per eV.

Run:  python benchmarks/reproduce_sundman_aucu.py
"""

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from landau import CompoundEnergyPhase
from landau.calculate import calc_phase_diagram
from landau.plot import get_phase_colors

JMOL = 96485.332  # J/mol per eV/atom
NS = 4            # sites per formula unit

# --- Table 1: compound-energy coefficients (J/mol formula) ------------------
u1, u2, u3, u4, u5, u6 = -7590, -9590, -9900, 10.32, 10.62, -1.565


def _G_Au3Cu(T):
    return (3 * u1 + 3 * u4 * T + 3 * u6 * T * np.log(T)) / NS / JMOL


def _G_Au2Cu2(T):
    return (4 * u2 + 4 * u5 * T + 4 * u6 * T * np.log(T)) / NS / JMOL


def _G_AuCu3(T):
    return (3 * u3 + 3 * u4 * T + 3 * u6 * T * np.log(T)) / NS / JMOL


# end-member energy depends only on the number of Cu among the four sublattices
_BY_NCU = {0: lambda T: 0.0, 1: _G_Au3Cu, 2: _G_Au2Cu2, 3: _G_AuCu3, 4: lambda T: 0.0}
_ENDMEMBERS = {tuple(int(b) for b in cfg): _BY_NCU[sum(cfg)] for cfg in np.ndindex(2, 2, 2, 2)}


def _excess(y, T):
    """Regular (eq 8) + reciprocal (eq 9) interactions, per atom (eV)."""
    # regular  L_{Au,Cu:*:*:*} = 3940 + 10.32 T, one copy per sublattice
    e_reg = (3940 + 10.32 * T) * np.sum(y * (1 - y))
    # reciprocal parameters, collapsed over the two non-interacting sublattices:
    #   L=-18T when they hold <=1 Cu, L=-15900-18T when both hold Cu, i.e.
    #   (1-y_r)y_r (1-y_s)y_s [ -18T - 15900 y_t y_u ]  over the 6 interacting pairs
    e_rec = 0.0
    for r, s in combinations(range(4), 2):
        t, u = (i for i in range(4) if i not in (r, s))
        e_rec += (1 - y[r]) * y[r] * (1 - y[s]) * y[s] * (-18 * T - 15900 * y[t] * y[u])
    return (e_reg + e_rec) / NS / JMOL


_COMMON = dict(site_multiplicities=(0.25, 0.25, 0.25, 0.25), endmember_energies=_ENDMEMBERS, excess=_excess)


def au_cu_phases():
    """Disordered A1 plus the three ordered superstructures as competing phases."""
    fcc = CompoundEnergyPhase(name="fcc", orderings=(), **_COMMON)  # disordered A1
    au3cu = CompoundEnergyPhase(name="Au3Cu", orderings=((1, 0, 0, 0),), include_disordered_seed=False, **_COMMON)
    aucu = CompoundEnergyPhase(name="AuCu", orderings=((1, 1, 0, 0),), include_disordered_seed=False, **_COMMON)
    aucu3 = CompoundEnergyPhase(name="AuCu3", orderings=((1, 1, 1, 0),), include_disordered_seed=False, **_COMMON)
    return [fcc, au3cu, aucu, aucu3]


def main():
    phases = au_cu_phases()

    # --- verification against the paper -------------------------------------
    print(f"AuCu (L1_0) end-member energy at 300 K: {_G_Au2Cu2(300) * JMOL:8.1f} J/mol-atom  (Fig 5 ~ -9000)")

    temperatures = np.arange(290, 720, 10.0)
    mu = np.linspace(-0.5, 0.5, 121)
    df = calc_phase_diagram(phases, temperatures, mu=mu, refine=False)

    # plot_phase_diagram's polygon renderer cannot represent the disordered matrix
    # pierced by the ordered-island domes (a multiply-connected region), so we draw
    # the calc_phase_diagram equilibria directly: each stable (c, T) grid point,
    # coloured by the phase that minimises the semi-grand potential there.
    stable = df.query("stable")
    colors = get_phase_colors([p.name for p in phases])

    plt.figure(figsize=(6.4, 4.7))
    for name, g in stable.groupby("phase"):
        plt.scatter(g["c"], g["T"], s=6, color=colors[name], label=name)
    plt.xlim(0, 1)
    plt.ylim(290, 720)
    plt.xlabel("Mole fraction Cu")
    plt.ylabel("Temperature (K)")
    plt.title("Au-Cu fcc order/disorder (Sundman et al. 1998)")
    plt.legend(markerscale=2, ncol=4, loc="upper center", fontsize=9)
    plt.tight_layout()
    out = "benchmarks/_aucu_order_disorder.png"
    plt.savefig(out, dpi=130)
    counts = stable.phase.value_counts().to_dict()
    print(f"wrote {out}  (stable grid points per phase: {counts})")


if __name__ == "__main__":
    main()
