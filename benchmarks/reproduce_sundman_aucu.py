"""Reproduce the Au-Cu order/disorder phase diagram of

    B. Sundman, S. G. Fries, W. A. Oates,
    "A thermodynamic assessment of the Au-Cu system",
    Calphad 22 (1998) 335-354,   Assessment I, Figure 3.

The fcc-based phases (disordered A1 and the ordered L1_0 / L1_2 superstructures)
are a single four-sublattice Compound Energy Formalism phase; ordering is an
internal (site-fraction) degree of freedom.  All parameters are given here as
code -- no database file is read.  Units: the CALPHAD parameters are per mole of
formula (4 sites) in J/mol; ``CompoundEnergyPhase`` works per atom in eV, so the
per-formula energies are divided by NS=4 sites and by 96485.33 J/mol per eV.

The order/disorder transitions are first order, so at fixed dmu the equilibrium
concentration jumps between the ordered plateau and the disordered branch.  We
locate those jumps along dmu at each temperature; the two compositions bracketing
a jump are the tie-line ends -- the boundaries of the ordered domes.

Run:  python benchmarks/reproduce_sundman_aucu.py
"""

import os

# small dense-array solves oversubscribe BLAS threads against the outer loop
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from landau import CompoundEnergyPhase

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


def au_cu_fcc():
    """The four-sublattice fcc CEF phase (disordered A1 + ordered L1_0/L1_2)."""
    return CompoundEnergyPhase(
        name="fcc",
        site_multiplicities=(0.25, 0.25, 0.25, 0.25),
        endmember_energies=_ENDMEMBERS,
        excess=_excess,
    )


def order_disorder_boundary(phase, temperatures, mu=np.linspace(-0.5, 0.5, 81), jump=0.02):
    """Return (c, T) points on the order/disorder boundaries.

    At each T the equilibrium c(dmu) jumps at every first-order transition; the
    two concentrations bracketing a jump larger than ``jump`` are boundary points.
    """
    cs, ts = [], []
    for T in temperatures:
        c = phase.concentration(float(T), mu)  # one f(c) build per T, argmin over all mu
        (idx,) = np.where(np.abs(np.diff(c)) > jump)
        for i in idx:
            cs.extend([c[i], c[i + 1]])
            ts.extend([T, T])
    return np.array(cs), np.array(ts)


def main():
    phase = au_cu_fcc()

    # --- verification against the paper -------------------------------------
    print(f"AuCu (L1_0) end-member energy at 300 K: {_G_Au2Cu2(300) * JMOL:8.1f} J/mol-atom  (Fig 5 ~ -9000)")

    temperatures = np.arange(280, 700, 20.0)
    c, T = order_disorder_boundary(phase, temperatures)

    plt.figure(figsize=(6, 4.5))
    plt.scatter(c, T, s=8, color="tab:blue")
    plt.xlim(0, 1)
    plt.ylim(250, 720)
    plt.xlabel("Mole fraction Cu")
    plt.ylabel("Temperature (K)")
    plt.title("Au-Cu fcc order/disorder (Sundman et al. 1998, Fig. 3)")
    for xc, lab in [(0.25, "L1$_2$"), (0.5, "L1$_0$"), (0.75, "L1$_2$")]:
        plt.text(xc, 300, lab, ha="center")
    plt.tight_layout()
    out = "benchmarks/_aucu_order_disorder.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(c)} boundary points over {len(temperatures)} isotherms)")


if __name__ == "__main__":
    main()
