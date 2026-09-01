"""``PhonopyQuasiHarmonicPhase`` on a real quasi-harmonic calculation.

The unit tests hold the phase against oracles written out from the statistics -- an
Einstein solid, a planted Vinet curve -- on hand-built spectra.  This runs the real
thing: fcc Cu, nine volumes, EMT forces, phonopy force constants and a 12^3 mesh, and
checks it against the two external references such a calculation has.

1. The mode sum is phonopy's own.  The phase resets the temperature on the caller's
   :class:`~phonopy.phonon.thermal_properties.ThermalProperties` and reads the result
   back, so the free energy must come back *identical* to what a freshly run
   :class:`~phonopy.Phonopy` reports, not close to it.

2. The volume minimisation reaches :class:`~phonopy.api_qha.PhonopyQHA`'s answer.  Both
   fit an equation of state through the same per-volume free energies and minimise it,
   but the fit is landau's own, so agreement is a real cross-check.

3. Routing the same ``G(T)`` through
   :class:`~landau.phases.TemperatureDependentLinePhase` -- how a quasi-harmonic solid
   reached landau before -- displaces a shallow solid-solid transition by the fit
   residual over the slope of the free-energy difference.  The crossing is built with
   ``d(dF)/dT = 0.0227 meV/atom/K``, the measured slope of the fcc-bcc difference in Ca,
   where one meV/atom of fit error is worth 44 K.

Needs the ``phonopy`` and ``ase`` extras; the ``test`` extra pulls in both.  Run with
``-s`` to print the measured residuals and shifts.
"""

import numpy as np
import pytest
import scipy.optimize as so
from numpy.testing import assert_allclose, assert_array_equal
from pyiron_snippets.import_alarm import ImportAlarm

from landau.interpolate import SGTE, PolyFit
from landau.phases import TemperatureDependentLinePhase

with ImportAlarm() as qha_alarm:
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from phonopy import Phonopy
    from phonopy.api_qha import PhonopyQHA
    from phonopy.physical_units import get_physical_units
    from phonopy.structure.atoms import PhonopyAtoms

    from landau.phases.quasiharmonic import PhonopyQuasiHarmonicPhase

pytestmark = pytest.mark.skipif(qha_alarm.message is not None, reason="phonopy and ase are not installed")


LATTICE_CONSTANTS = np.linspace(3.45, 3.90, 9)
MESH = [12, 12, 12]
TEMPERATURES = np.arange(0.0, 1201.0, 25.0)
CROSSINGS = (300.0, 600.0, 900.0)
# d(F_fcc - F_bcc)/dT for Ca, the shallow solid-solid crossing that motivated the phase
CA_SLOPE = 0.0227e-3  # eV/atom/K


def quasi_harmonic_copper(a):
    """One sampled volume: EMT static energy, phonopy force constants and a mesh.

    Thermal properties are the caller's to run -- that is where the mesh, the statistics
    and the mode cutoff are chosen -- so this stops at the mesh and the tests below run
    them with the statistics they are about.
    """
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
    return phonon, energy


@pytest.fixture(scope="module")
def copper():
    """Nine volumes of fcc Cu as live Phonopy objects, with their EMT static energies."""
    return [quasi_harmonic_copper(a) for a in LATTICE_CONSTANTS]


def phonopy_thermal_properties(phonon, classical, temperatures=TEMPERATURES):
    """phonopy's own free energy, entropy and heat capacity over ``temperatures``.

    Read off the ``thermal_properties`` tuple rather than the same-named attributes, which
    phonopy 3 does not carry.  Each call replaces ``phonon.thermal_properties`` with a new
    object, so a phase already built keeps the one it was given.
    """
    phonon.run_thermal_properties(temperatures=temperatures, classical=classical)
    _, free_energy, entropy, heat_capacity = phonon.thermal_properties.thermal_properties
    return free_energy, entropy, heat_capacity


def copper_phase(copper, classical=False):
    """A phase over those volumes, with the statistics the caller picks when running.

    The conventional cell holds four atoms and reduces to one primitive atom, so this is
    also the case where the phonons and the static energy carry different atom counts.
    """
    for phonon, _ in copper:
        phonon.run_thermal_properties(temperatures=TEMPERATURES, classical=classical)
    phase = PhonopyQuasiHarmonicPhase.from_phonopy(
        "Cu", 0.0, [p for p, _ in copper], [e for _, e in copper]
    )
    assert phase.atoms_per_cell == 4 and phase.atoms_per_primitive_cell == 1
    # the volumes are sampled in ascending order and all are stable, so the rows of
    # volume_free_energies line up with the inputs as given
    assert_allclose(phase.sampled_volumes, phase.volumes_per_atom, rtol=0, atol=0)
    return phase


@pytest.mark.parametrize("classical", [False, True])
def test_mode_sum_is_phonopys_own(copper, classical):
    phase = copper_phase(copper, classical)
    to_ev = 1 / get_physical_units().EvTokJmol
    static = np.array(phase.energies_per_atom)
    for T in (0.0, 300.0, 900.0, 1200.0):
        reference = np.array(
            [phonopy_thermal_properties(phonon, classical, [T])[0][0] for phonon, _ in copper]
        )
        # identical, not close: the phase resets the temperature on phonopy's own object
        # and reads its own kernel back, so any difference at all means something diverged
        assert_array_equal(
            phase.volume_free_energies(T),
            static + reference * to_ev / phase.atoms_per_primitive_cell,
        )


@pytest.mark.parametrize("classical", [False, True])
def test_volume_minimisation_matches_phonopy_qha(copper, classical):
    per_volume = [phonopy_thermal_properties(phonon, classical) for phonon, _ in copper]
    # PhonopyQHA wants each quantity shaped (n_temperatures, n_volumes)
    free_energy, entropy, heat_capacity = (np.array(q).T for q in zip(*per_volume, strict=True))
    phase = copper_phase(copper, classical)
    per_primitive = phase.atoms_per_primitive_cell
    # PhonopyQHA is deprecated in phonopy 4 in favour of an API phonopy 3 does not carry;
    # it is present across the whole supported range, which the replacement is not
    qha = PhonopyQHA(
        volumes=np.array(phase.volumes_per_atom) * per_primitive,
        electronic_energies=np.array(phase.energies_per_atom) * per_primitive,
        temperatures=TEMPERATURES,
        free_energy=free_energy,
        entropy=entropy,
        cv=heat_capacity,
        eos="vinet",
    )
    reference_G = np.array(qha.gibbs_temperature) / per_primitive
    reference_V = np.array(qha.volume_temperature) / per_primitive
    # PhonopyQHA drops the temperatures where its own differences run off the end
    T = TEMPERATURES[: len(reference_G)]
    assert len(T) > 0.9 * len(TEMPERATURES)

    G = phase.line_free_energy(T)
    V = phase.equilibrium_volume(T)
    assert np.isfinite(G).all()
    # two independent equation-of-state fits and minimisations of the same energies
    assert np.abs(G - reference_G).max() < 1e-8
    assert np.abs(V - reference_V).max() / reference_V.mean() < 1e-7
    # and the answer is not a constant the tolerances would also accept
    assert V[-1] - V[0] > 0.05 * V[0]


# Bands the fit cost falls in, per interpolator: the worst residual anywhere on
# TEMPERATURES, and how far the crossing moves.  Wide enough not to track a phonopy
# release, tight enough that the two interpolators cannot swap places.
FIT_CASES = [
    # interpolator, worst residual (meV/atom), |shift| at each crossing (K)
    pytest.param(SGTE(3), (10.0, 30.0), (40.0, 400.0), None, id="SGTE-3"),
    pytest.param(PolyFit(8), (0.1, 1.0), (0.3, 15.0), 0.15, id="PolyFit-8"),
]


@pytest.mark.parametrize("interpolator,residual_band,shift_band,predicted_rtol", FIT_CASES)
def test_fit_residual_displaces_a_shallow_crossing(copper, interpolator, residual_band, shift_band, predicted_rtol):
    """What sampling this curve onto a grid and fitting it costs a solid-solid transition.

    The competitor is the same solid tilted by a constant slope, so the free-energy
    *difference* is exactly ``CA_SLOPE * (T - crossing)``: a straight line through zero
    at ``crossing``.  Two real solids behave like this -- their curvatures very nearly
    cancel -- which is what makes such a crossing sensitive.  (A straight ``LinePhase``
    competitor would not model it: its curvature does not cancel the solid's, so the
    difference bends over within ~50 K and the crossing stops being shallow.)

    Routing the solid through a fit adds the residual ``r(T)`` to that difference, so the
    transition moves to where ``r(T) + CA_SLOPE * (T - crossing) = 0``, i.e. by about
    ``-r(crossing) / CA_SLOPE``.  Exact while the shift is small enough that ``r`` barely
    changes across it, which is why only the accurate fit is held to it.
    """
    phase = copper_phase(copper)
    sampled = phase.line_free_energy(TEMPERATURES)
    fitted = TemperatureDependentLinePhase(
        "fitted",
        fixed_concentration=0.0,
        temperatures=TEMPERATURES,
        free_energies=sampled,
        interpolator=interpolator,
    )

    def residual(T):
        return float(fitted.line_free_energy(T) - phase.line_free_energy(T))

    worst = float(np.abs(fitted.line_free_energy(TEMPERATURES) - sampled).max())
    assert residual_band[0] < worst * 1e3 < residual_band[1]

    for crossing in CROSSINGS:
        local = residual(crossing)
        moved = so.brentq(lambda T, x=crossing: residual(T) + CA_SLOPE * (T - x), 50.0, 1200.0)
        shift, predicted = moved - crossing, -local / CA_SLOPE
        print(
            f"{type(interpolator).__name__}({interpolator.nparam}) worst {worst * 1e3:6.2f}m  "
            f"T_cross {crossing:4.0f}  local {local * 1e3:+7.3f}m  shift {shift:+7.1f}K  "
            f"predicted {predicted:+7.1f}K"
        )
        # the fit displaces the crossing, in the direction the local residual sets
        assert np.sign(shift) == np.sign(predicted)
        assert shift_band[0] < abs(shift) < shift_band[1]
        if predicted_rtol is not None:
            assert abs(shift - predicted) < predicted_rtol * abs(predicted)
