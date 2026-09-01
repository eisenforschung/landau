"""What the interpolation step costs a quasi-harmonic solid.

Builds a real quasi-harmonic calculation for fcc Cu -- nine volumes, EMT forces,
phonopy force constants and mesh -- and reports three things:

1. ``PhononSpectrum`` against phonopy's own ``ThermalProperties`` driven from the live
   ``Phonopy`` object, both statistics.  These must be identical, not close: the spectrum
   is presented to the same class through the attributes it reads off a mesh.

2. ``PhonopyQuasiHarmonicPhase`` against ``phonopy.api_qha.PhonopyQHA`` fed the same
   per-volume free energies, each doing its own equation-of-state fit and minimisation.

3. The same ``G(T)`` routed through ``TemperatureDependentLinePhase``, which is how a
   quasi-harmonic solid reached landau before: the fit residual, and what that residual
   does to a shallow solid-solid transition.  The crossing is set up with
   ``d(dF)/dT = 0.0227 meV/atom/K``, the measured slope of the fcc-bcc difference in Ca,
   where one meV/atom of fit error is worth 44 K.

Needs the ``phonopy`` and ``ase`` extras.  Run from the repo root:

    python benchmarks/qha_phase_vs_fit.py

Measured 2026-09-01 (phonopy 4.4.0, ase 3.29.0, 9 volumes, 12^3 mesh, T = 0-1200 K):

    F_vib vs phonopy ThermalProperties   quantum   0.000e+00 eV/atom
                                         classical 0.000e+00 eV/atom
    G(T)  vs PhonopyQHA                  quantum   5.9e-10 eV/atom  (dV/V 2.4e-09)
                                         classical 9.8e-10 eV/atom  (dV/V 3.9e-09)

    interpolator  max resid  T_cross  local resid     shift  predicted
    SGTE(3)          17.13m      300       2.473m   -192.7K    -109.0K
    SGTE(3)          17.13m      600      -1.999m   +123.4K     +88.1K
    SGTE(3)          17.13m      900      -2.342m    +66.5K    +103.2K
    PolyFit(8)        0.38m      300       0.119m     -5.1K      -5.3K
    PolyFit(8)        0.38m      600      -0.059m     +2.5K      +2.6K
    PolyFit(8)        0.38m      900      -0.016m     +0.7K      +0.7K

Two meV/atom of residual is worth 60-190 K here. What matters is the residual AT the
crossing, not the worst residual anywhere, and the shift is -r(T_cross) / slope: exact
for PolyFit, where the shift is small enough that r barely changes across it, and only
indicative for SGTE, where the shift is large enough that it does.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as so
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from phonopy import Phonopy
from phonopy.api_qha import PhonopyQHA
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

from landau.interpolate import SGTE, PolyFit
from landau.phases import TemperatureDependentLinePhase
from landau.phases.quasiharmonic import PhononSpectrum, PhonopyQuasiHarmonicPhase

LATTICE_CONSTANTS = np.linspace(3.45, 3.90, 9)
MESH = [12, 12, 12]
TEMPERATURES = np.arange(0.0, 1201.0, 25.0)
# d(F_fcc - F_bcc)/dT for Ca, the shallow solid-solid crossing that motivated this
CA_SLOPE = 0.0227e-3  # eV/atom/K


def quasi_harmonic_copper(a: float) -> tuple[Phonopy, float, int]:
    """One volume: EMT static energy, phonopy force constants and a sampled mesh."""
    atoms = bulk("Cu", "fcc", a=a, cubic=True)
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    phonon = Phonopy(
        PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.cell,
            scaled_positions=atoms.get_scaled_positions(),
        ),
        supercell_matrix=np.eye(3) * 2,
        primitive_matrix="auto",
    )
    phonon.generate_displacements(distance=0.01)
    forces = []
    for supercell in phonon.supercells_with_displacements:
        image = Atoms(
            symbols=supercell.symbols,
            cell=supercell.cell,
            scaled_positions=supercell.scaled_positions,
            pbc=True,
        )
        image.calc = EMT()
        forces.append(image.get_forces())
    phonon.forces = np.array(forces)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.run_mesh(MESH, is_gamma_center=True)
    return phonon, energy, len(atoms)


def phonopy_free_energies(phonon: Phonopy, classical: bool) -> np.ndarray:
    """phonopy's own vibrational free energy in kJ/mol per primitive cell."""
    phonon.run_thermal_properties(temperatures=TEMPERATURES, classical=classical)
    return phonon.thermal_properties.thermal_properties[1]


def main() -> None:
    volumes = [quasi_harmonic_copper(a) for a in LATTICE_CONSTANTS]
    spectra = tuple(PhononSpectrum.from_phonopy(p, energy=e, atoms_per_cell=n) for p, e, n in volumes)
    per_primitive = spectra[0].atoms_per_primitive_cell
    per_cell = spectra[0].atoms_per_cell
    to_ev = 1 / get_physical_units().EvTokJmol
    print(f"{len(spectra)} volumes, {per_cell} atoms per cell, {per_primitive} per primitive cell")

    # 1. the mode sum is phonopy's own
    for classical in (False, True):
        worst = 0.0
        for (phonon, _, _), spectrum in zip(volumes, spectra, strict=True):
            reference = phonopy_free_energies(phonon, classical) * to_ev / per_primitive
            mine = spectrum.vibrational_free_energy(TEMPERATURES, classical=classical)
            worst = max(worst, float(np.abs(mine - reference).max()))
        print(f"F_vib vs phonopy ThermalProperties  classical={classical!s:5s}  {worst:.3e} eV/atom")

    # 2. the volume minimisation against PhonopyQHA on the same free energies
    for classical in (False, True):
        free_energy, entropy, heat_capacity = (
            np.array(
                [
                    getattr(phonopy_run(phonon, classical), name)
                    for phonon, _, _ in volumes
                ]
            ).T
            for name in ("free_energy", "entropy", "heat_capacity")
        )
        primitive_volumes = np.array([s.volume_per_atom for s in spectra]) * per_primitive
        primitive_energies = np.array([s.static_energy_per_atom for s in spectra]) * per_primitive
        qha = PhonopyQHA(
            volumes=primitive_volumes,
            electronic_energies=primitive_energies,
            temperatures=TEMPERATURES,
            free_energy=free_energy,
            entropy=entropy,
            cv=heat_capacity,
            eos="vinet",
        )
        reference_G = np.array(qha.gibbs_temperature) / per_primitive
        reference_V = np.array(qha.volume_temperature) / per_primitive
        phase = PhonopyQuasiHarmonicPhase("Cu", 0.0, spectra=spectra, classical=classical)
        n = len(reference_G)
        mine_G = phase.line_free_energy(TEMPERATURES[:n])
        mine_V = phase.equilibrium_volume(TEMPERATURES[:n])
        print(
            f"G(T)  vs PhonopyQHA                 classical={classical!s:5s}  "
            f"{np.abs(mine_G - reference_G).max():.3e} eV/atom  "
            f"dV/V {np.abs(mine_V - reference_V).max() / reference_V.mean():.3e}"
        )

    # 3. what routing the same curve through a fit costs a shallow transition
    #
    # The competitor is this same solid tilted by a constant slope, so the free-energy
    # DIFFERENCE is exactly CA_SLOPE * (T - crossing): a straight line through zero at
    # `crossing` with the slope measured for the Ca fcc-bcc pair.  Two real solids behave
    # like this -- their curvatures very nearly cancel and the difference is close to
    # linear, which is precisely what makes such a crossing so sensitive.  (A straight
    # LinePhase competitor would NOT model it: its curvature does not cancel the solid's,
    # so the difference bends over within ~50 K and the crossing stops being shallow.)
    #
    # Routing the solid through a fit adds the residual r(T) to that difference, so the
    # transition moves to where r(T) + CA_SLOPE * (T - crossing) = 0, i.e. by about
    # -r(crossing) / CA_SLOPE.
    phase = PhonopyQuasiHarmonicPhase("Cu", 0.0, spectra=spectra)
    sampled = phase.line_free_energy(TEMPERATURES)
    print(f"\nfree-energy difference slope d(dF)/dT = {CA_SLOPE * 1e3:.4f} meV/atom/K (Ca fcc-bcc)")
    print(f"{'interpolator':<12s} {'max resid':>10s} {'T_cross':>8s} {'local resid':>12s} {'shift':>9s} {'predicted':>10s}")
    for interpolator in (SGTE(3), PolyFit(8)):
        fitted = TemperatureDependentLinePhase(
            "fitted",
            fixed_concentration=0.0,
            temperatures=TEMPERATURES,
            free_energies=sampled,
            interpolator=interpolator,
        )
        label = f"{type(interpolator).__name__}({interpolator.nparam})"
        worst = float(np.abs(fitted.line_free_energy(TEMPERATURES) - sampled).max())

        def residual(T, fit=fitted):
            return float(fit.line_free_energy(T) - phase.line_free_energy(T))

        for crossing in (300.0, 600.0, 900.0):
            local = residual(crossing)
            shifted = so.brentq(
                lambda T, fit=fitted, x=crossing: residual(T, fit) + CA_SLOPE * (T - x),
                50.0,
                1200.0,
            )
            print(
                f"{label:<12s} {worst * 1e3:9.2f}m {crossing:8.0f} {local * 1e3:11.3f}m "
                f"{shifted - crossing:+8.1f}K {-local / CA_SLOPE:+9.1f}K"
            )


def phonopy_run(phonon: Phonopy, classical: bool):
    """phonopy's ``ThermalProperties`` object after running the shared temperature list."""
    phonon.run_thermal_properties(temperatures=TEMPERATURES, classical=classical)
    return phonon.thermal_properties


if __name__ == "__main__":
    main()
