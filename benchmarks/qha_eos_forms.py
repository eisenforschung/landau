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

    fit residual over all nine volumes, max |r| in meV/atom

        T     PolyFit(8)   vinet   birch_murnaghan   murnaghan
        0        0.013     0.293        0.419          0.715
      600        0.053     1.668        1.643          2.292
     1200        0.105     4.258        2.955          4.851

    fit residual at five volumes, where the cap leaves PolyFit four parameters too

        span / T        PolyFit   vinet   birch_murnaghan   murnaghan
        a +-6%,    0     1.147    0.302        0.412          0.678
        a +-6%, 1200     0.795    3.245        2.291          3.739
        a +-3%, 1200     0.174    0.544        0.429          0.611

    leave-one-out error over nine volumes, max |error| in meV/atom

        T     PolyFit(8)   vinet   birch_murnaghan   murnaghan
        0        2.65      1.50         2.21           4.07
      600       10.53      8.16         7.56          13.93
     1200       18.45     26.59        13.17          34.72

The default clears a millielectronvolt by a wide margin where there are volumes to spend
on it: forty times Vinet's accuracy at 1200 K over nine volumes, and under 0.11 meV/atom
throughout.  Two things temper that.  The parameter cap means five volumes buy a cubic,
where the advantage over the closed forms survives only at high temperature and reverses
at 0 K.  And the leave-one-out column does not follow the residual: predicting a held-out
volume, Birch-Murnaghan is the better shape at every temperature except the hottest, so
the polynomial is buying interpolation accuracy between the sampled volumes rather than a
better model of F(V).  That is the right trade here, since the minimum being reported sits
between sampled volumes and never outside them, but it is not the same claim.

The remaining table is what a phase diagram sees -- how far the reported free energy moves
when the form changes.  Across the three closed forms alone it is 0.13 meV/atom at 0 K
rising to 0.98 at 1200 K on the wide five-volume set, and 0.03 to 0.13 on the narrow one.
Adding the default widens it, because it disagrees with all three: it is closer to the
computed points than any of them.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, "tests/integration")

from test_qha_vs_phonopy import LATTICE_CONSTANTS, quasi_harmonic_copper

from landau.interpolate import PolyFit
from landau.phases.quasiharmonic import PhonopyQuasiHarmonicPhase

FORMS = (PolyFit(8), "vinet", "birch_murnaghan", "murnaghan")
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
        f"Cu-{eos_label(eos)}",
        0.0,
        thermal_properties=[p.thermal_properties for p, _ in chosen],
        volumes=[p.unitcell.volume for p, _ in chosen],
        energies=[e for _, e in chosen],
        atoms_per_cell=4,
        atoms_per_primitive_cell=1,
        eos=eos,
    )


def eos_label(eos):
    return eos if isinstance(eos, str) else f"{type(eos).__name__}({eos.nparam})"


def residuals(p, T):
    """How far the fit misses each sampled volume, in eV/atom."""
    return p.helmholtz_free_energies(T) - np.asarray(p._fit(T).curve(p.sampled_volumes))


def report_span(volumes, indices, label):
    p0 = phase(volumes, indices, FORMS[0])
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
            print(f"{T:6.0f} {eos_label(eos):>16} {np.abs(r).max() * 1e3:11.3f} "
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
                predicted = without._fit(T).curve(np.array([sampled[held]]))
                errors.append(truth[held] - float(np.atleast_1d(predicted)[0]))
            errors = np.array(errors)
            print(f"{T:6.0f} {eos_label(eos):>16} {np.abs(errors).max() * 1e3:13.3f} "
                  f"{np.sqrt((errors**2).mean()) * 1e3:8.3f}")


def main():
    volumes = build()
    report_span(volumes, list(range(len(LATTICE_CONSTANTS))), "all nine")
    report_span(volumes, WIDE, "wide")
    report_span(volumes, NARROW, "narrow")
    report_leave_one_out(volumes)


if __name__ == "__main__":
    main()
