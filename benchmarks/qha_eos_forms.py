"""Which equation of state fits quasi-harmonic data best, and does it matter?

`PhonopyQuasiHarmonicPhase` minimises a fitted equation of state over volume, so the fit
is a second approximation sitting underneath the interpolation the phase removes.  This
measures how big it is, on the fcc Cu / EMT calculation the integration test builds, and
whether the choice of functional form is the lever for it.

Three things are reported per form and temperature:

1. the fit residual at the sampled volumes -- what ``check_equation_of_state(plot_error=
   True)`` draws;
2. the leave-one-out error, refitting without each volume and predicting it, which is the
   honest test of whether the form has the right shape rather than enough freedom;
3. the spread of the reported ``line_free_energy`` across the three forms, which is what
   actually reaches a phase diagram.

Two volume sets of the *same size* are compared so that span is separated from degrees of
freedom: five volumes over a = 3.45-3.90 A (+-6%) against five over 3.56-3.79 A (+-3%).
Four parameters through five points leaves one degree of freedom either way.

Needs the ``phonopy`` and ``ase`` extras.  Run from the repo root, about a minute:

    python benchmarks/qha_eos_forms.py

Measured 2026-09-02 (phonopy 4.4.0, ase 3.29.0, 12^3 mesh):

    leave-one-out error over nine volumes, max |error| in meV/atom

        T       vinet   birch_murnaghan   murnaghan
        0        1.50        2.21            4.07
      600        8.16        7.56           13.93
     1200       26.59       13.17           34.72

    fit residual at five sampled volumes, max |r| in meV/atom at 1200 K

        span            vinet   birch_murnaghan   murnaghan
        a +-6%           3.25        2.29            3.74
        a +-3%           0.54        0.43            0.61

    spread of the reported free energy across the three forms, meV/atom

        span             0 K    600 K   1200 K
        a +-6%          0.13     0.65     0.98
        a +-3%          0.03     0.02     0.13

Birch-Murnaghan is the better shape once the solid is hot -- half Vinet's leave-one-out
error at 1200 K -- while Vinet is the better one cold, and Murnaghan is worse than both
everywhere.  But the form is the second-order knob: at fixed point count, halving the
sampled span cuts the residual by six, where changing form buys thirty percent.  What
reaches a phase diagram is the last table, and at the Ca fcc-bcc slope of
0.0227 meV/atom/K a spread of 0.98 meV/atom is 43 K of transition temperature while
0.13 meV/atom is 6 K.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, "tests/integration")

from test_qha_vs_phonopy import LATTICE_CONSTANTS, quasi_harmonic_copper

from landau.phases.quasiharmonic import PhonopyQuasiHarmonicPhase, _eos_curve

FORMS = ("vinet", "birch_murnaghan", "murnaghan")
TEMPERATURES = (0.0, 300.0, 600.0, 900.0, 1200.0)
WIDE = [0, 2, 4, 6, 8]  # a = 3.45-3.90 A
NARROW = [2, 3, 4, 5, 6]  # a = 3.56-3.79 A


def build():
    """The nine sampled volumes, with thermal properties run once."""
    volumes = [quasi_harmonic_copper(a) for a in LATTICE_CONSTANTS]
    for phonon, _ in volumes:
        phonon.run_thermal_properties(temperatures=[0.0])
    return volumes


def phase(volumes, indices, eos):
    chosen = [volumes[i] for i in indices]
    return PhonopyQuasiHarmonicPhase(
        f"Cu-{eos}",
        0.0,
        thermal_properties=[p.thermal_properties for p, _ in chosen],
        volumes=[p.unitcell.volume for p, _ in chosen],
        energies=[e for _, e in chosen],
        atoms_per_cell=4,
        atoms_per_primitive_cell=1,
        eos=eos,
    )


def residuals(p, T):
    """How far the fit misses each sampled volume, in eV/atom."""
    return p.helmholtz_free_energies(T) - _eos_curve(p.eos, p.sampled_volumes, p.eos_parameters(T))


def report_span(volumes, indices, label):
    p0 = phase(volumes, indices, "vinet")
    v = p0.sampled_volumes
    print(f"\n=== {label}: {len(indices)} volumes, a = {LATTICE_CONSTANTS[indices[0]]:.2f}-"
          f"{LATTICE_CONSTANTS[indices[-1]]:.2f} A, V = {v[0]:.2f}-{v[-1]:.2f} A^3/atom ===")
    print(f"{'T':>6} {'form':>16} {'max|r| meV':>11} {'rms meV':>8} {'F_min eV':>12} {'V_min':>8}")
    for T in TEMPERATURES:
        minima = {}
        for eos in FORMS:
            p = phase(volumes, indices, eos)
            r = residuals(p, T)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                minima[eos] = (p.line_free_energy(T), p.equilibrium_volume(T))
            print(f"{T:6.0f} {eos:>16} {np.abs(r).max() * 1e3:11.3f} "
                  f"{np.sqrt((r**2).mean()) * 1e3:8.3f} {minima[eos][0]:12.6f} {minima[eos][1]:8.4f}")
        free = [f for f, _ in minima.values()]
        print(f"{'':6} {'spread of F_min':>16} {(max(free) - min(free)) * 1e3:11.4f} meV/atom")


def report_leave_one_out(volumes):
    """Refit without each volume and predict it: shape, not freedom."""
    print(f"\n=== leave-one-out over all {len(volumes)} volumes ===")
    print(f"{'T':>6} {'form':>16} {'max|err| meV':>13} {'rms meV':>8}")
    everything = list(range(len(volumes)))
    for T in TEMPERATURES:
        for eos in FORMS:
            full = phase(volumes, everything, eos)
            sampled, truth = full.sampled_volumes, full.helmholtz_free_energies(T)
            errors = []
            for held in everything:
                without = phase(volumes, [i for i in everything if i != held], eos)
                predicted = _eos_curve(eos, np.array([sampled[held]]), without.eos_parameters(T))
                errors.append(truth[held] - float(np.atleast_1d(predicted)[0]))
            errors = np.array(errors)
            print(f"{T:6.0f} {eos:>16} {np.abs(errors).max() * 1e3:13.3f} "
                  f"{np.sqrt((errors**2).mean()) * 1e3:8.3f}")


def main():
    volumes = build()
    report_span(volumes, WIDE, "wide")
    report_span(volumes, NARROW, "narrow")
    report_leave_one_out(volumes)


if __name__ == "__main__":
    main()
