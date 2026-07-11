"""Reproduce the Au-Cu phase diagram of

    B. Sundman, S. G. Fries, W. A. Oates,
    "A thermodynamic assessment of the Au-Cu system",
    Calphad 22 (1998) 335-354.

This reproduces the paper's **Assessment I** (Table 1, eqs 6-9): the fcc-based
phases are modelled with the four-sublattice Compound Energy Formalism, capturing
the ordered L1_0 (AuCu) and L1_2 (Au3Cu, AuCu3) superstructures inside the
disordered A1 matrix (Figs. 1, 3), together with the liquid.  The paper also gives
an Assessment II that refits the liquid and the disordered fcc as plain
substitutional solutions (eq 10) and *drops the ordering entirely* -- it is a
different fit on its own parameters and has no ordered phases, so it is not a
side-by-side variant of Assessment I and is not drawn here.

The liquid (a substitutional solution, eqs 1-3/6) is built through landau's
Calphad-surface path -- ``Surface2DInterpolatingPhase`` with a
``CalphadSurface2DInterpolator`` (SGTE terminals + polynomial-in-T Redlich-Kister)
-- rather than a bespoke phase; the four-sublattice CEF phases are
``CompoundEnergyPhase`` instances.  All parameters are given as code, no database
file is read.

Units: the four-sublattice compound energies and interaction parameters are per
mole of formula (4 sites) in J/mol, so the per-formula ``G`` is divided by NS=4
sites; the liquid is per mole of atoms (1 site).  Everything is converted to
eV/atom by dividing by 96485.33 J/mol.

Run:  python benchmarks/reproduce_sundman_aucu.py
"""

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from landau import CompoundEnergyPhase
from landau.phases.compoundenergy import Endmember, RegularSolutionExcess, CallableExcess
from landau.phases import Surface2DInterpolatingPhase, TemperatureDependentLinePhase, S
from landau.interpolate import CalphadSurface2DInterpolator, SGTE
from landau.calculate import calc_phase_diagram
from landau.plot import plot_phase_diagram
from landau.refine import ClausiusClapeyronRefiner, DelaunayTripleRefiner

JMOL = 96485.332  # J/mol per eV/atom
NS = 4            # sites per fcc formula unit
_TMIN, _TMAX = 300.0, 1450.0  # diagram / surface-fit temperature span


# --- substitutional solutions (liquid, Assessment-II fcc) via the Calphad surface ---
# pure-element liquid lattice stabilities relative to fcc (SGTE, Dinsdale 1991), per
# atom (eV); zero at the element melting points (Au 1337.33 K, Cu 1357.77 K)


def _G_liq_Au(T):
    return (12552.45 - 9.385866 * T) / JMOL


def _G_liq_Cu(T):
    return (12964.735 - 9.511904 * T - 5.83932e-21 * T**7) / JMOL


def _calphad_solution(name, g_a, g_b, rk_coeffs, ncomp=9, nT=40):
    """A substitutional Redlich-Kister solution through landau's Calphad-surface path.

    Rather than re-rolling the RK evaluation into a bespoke phase, we sample
    ``f(c, T) = (1-c) g_a + c g_b + c(1-c) Σ_v L_v (1-2c)^v - T S(c)`` on a ``(c, T)``
    grid and hand the line-phase samples to ``Surface2DInterpolatingPhase`` with a
    ``CalphadSurface2DInterpolator`` (SGTE terminals + polynomial-in-T Redlich-Kister)
    -- the library's CALPHAD solution-phase representation.  ``g_a``/``g_b`` are the
    pure-element lattice stabilities (per atom, eV) and ``rk_coeffs`` the ``L_v(T)``
    callables (J/mol).
    """
    Tsweep = np.linspace(_TMIN, _TMAX, nT)
    lines = []
    for ci in np.linspace(0.0, 1.0, ncomp):
        d = 1 - 2 * ci
        excess = ci * (1 - ci) * sum(L(Tsweep) * d**v for v, L in enumerate(rk_coeffs)) / JMOL
        f = (1 - ci) * g_a(Tsweep) + ci * g_b(Tsweep) + excess - Tsweep * S(np.array(ci))
        lines.append(TemperatureDependentLinePhase(name, float(ci), Tsweep, f, interpolator=SGTE(3)))
    return Surface2DInterpolatingPhase(
        name, lines, concentration_range=(0.0, 1.0), add_entropy=False,
        temperature_range=(_TMIN, _TMAX), surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=3),
    )


def _liquid():
    """The liquid, a substitutional solution (Assessment I, eqs 1-3, 6)."""
    return _calphad_solution("liquid", _G_liq_Au, _G_liq_Cu,
                             (lambda T: -27900 - T, lambda T: 4730.0 + 0 * T, lambda T: 3500 + 3.5 * T))


# --- Assessment I: four-sublattice CEF fcc (Table 1, eqs 7-9) -----------------
_u1, _u2, _u3, _u4, _u5, _u6 = -7590, -9590, -9900, 10.32, 10.62, -1.565


def _G_Au3Cu(T):
    return (3 * _u1 + 3 * _u4 * T + 3 * _u6 * T * np.log(T)) / NS / JMOL


def _G_Au2Cu2(T):
    return (4 * _u2 + 4 * _u5 * T + 4 * _u6 * T * np.log(T)) / NS / JMOL


def _G_AuCu3(T):
    return (3 * _u3 + 3 * _u4 * T + 3 * _u6 * T * np.log(T)) / NS / JMOL


_BY_NCU = {0: lambda T: 0.0, 1: _G_Au3Cu, 2: _G_Au2Cu2, 3: _G_AuCu3, 4: lambda T: 0.0}
_ENDMEMBERS = tuple(
    Endmember(tuple(int(b) for b in cfg), _BY_NCU[sum(cfg)]) for cfg in np.ndindex(2, 2, 2, 2)
)


def _reciprocal_cef(y, T):
    """The reciprocal (eq 9) interaction, per atom (eV); broadcasts over y."""
    y = np.asarray(y, dtype=float)
    e_rec = 0.0
    for r, s in combinations(range(4), 2):
        t, u = (i for i in range(4) if i not in (r, s))
        e_rec = e_rec + (1 - y[..., r]) * y[..., r] * (1 - y[..., s]) * y[..., s] * (-18 * T - 15900 * y[..., t] * y[..., u])
    return e_rec / NS / JMOL


# Regular solution (eq 8, over all four sublattices) with an analytic gradient + the
# reciprocal term (eq 9, coupling all four) as a callable, summed by the phase; both
# per atom (eV).
_ALL = (0, 1, 2, 3)
_EXCESS = (
    RegularSolutionExcess(_ALL, lambda T: (3940 + 10.32 * T) / NS / JMOL),
    CallableExcess(_ALL, _reciprocal_cef),
)
_CEF = dict(site_multiplicities=(0.25, 0.25, 0.25, 0.25), endmembers=_ENDMEMBERS, excess=_EXCESS)


def _ordered_fcc_phases():
    """The three ordered fcc superstructures as basin-pinned CEF phases (shared energetics)."""
    return [
        CompoundEnergyPhase(name="Au3Cu", orderings=((1, 0, 0, 0),), include_disordered_seed=False, **_CEF),
        CompoundEnergyPhase(name="AuCu", orderings=((1, 1, 0, 0),), include_disordered_seed=False, **_CEF),
        CompoundEnergyPhase(name="AuCu3", orderings=((1, 1, 1, 0),), include_disordered_seed=False, **_CEF),
    ]


def assessment_i_phases():
    """Liquid + the CEF disordered A1 fcc + the three ordered superstructures."""
    return [_liquid(), CompoundEnergyPhase(name="fcc", orderings=(), **_CEF), *_ordered_fcc_phases()]


def _draw(ax, phases, title):
    # Coarse grid, dense in T where the ordered domes live, sparse over the liquid;
    # the grid solve is vectorised over the whole (T, dmu) mesh in one batch, and the
    # triple-point + Clausius-Clapeyron refiners locate the invariants and trace the
    # two-phase boundaries from it. The Clausius-Clapeyron step is loosened
    # (dT_max/dc_max) so the trace is sampled at roughly the Delaunay density rather
    # than its dense default, and the refinement probes warm-start each SCF from the
    # nearest cached grid solution. The boundary tracing (not the grid) dominates the
    # wall-clock; a single assessment refines in a few tens of seconds.
    temperatures = np.concatenate([np.arange(300, 720, 30.0), np.arange(760, 1420, 60.0)])
    mu = np.linspace(-0.7, 0.7, 31)
    refine = [DelaunayTripleRefiner(), ClausiusClapeyronRefiner(dT_max=40, dc_max=0.05, max_steps=120)]
    df = calc_phase_diagram(phases, temperatures, mu=mu, refine=refine)
    # Default poly_method falls through to segment-fasttsp when the fast-tsp extra
    # is installed (segment-tsp / concave otherwise). The TSP tour walks the border
    # points in order, so the dome caps come out smooth instead of the spiky rings
    # the plain segment stitcher produced at the ordered-phase apices.
    plot_phase_diagram(df, triplepoints=True, legend=True, inline_legend=False, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(300, 1420)
    ax.set_xlabel("Mole fraction Cu")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    return df.query("stable").phase.value_counts().to_dict()


def main():
    print(f"AuCu (L1_0) end-member energy at 300 K: {_G_Au2Cu2(300) * JMOL:8.1f} J/mol-atom  (Fig 5 ~ -9000)")
    print(f"Au melting point (Gliq = Gfcc): {12552.45 / 9.385866:7.1f} K  (paper 1337.33)")

    fig, ax = plt.subplots(figsize=(6, 5))
    c1 = _draw(ax, assessment_i_phases(),
               "Au-Cu Assessment I (Sundman, Fries & Oates 1998)")
    fig.tight_layout()
    out = "benchmarks/_aucu_phase_diagram.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"  Assessment I stable points: {c1}")


if __name__ == "__main__":
    main()
