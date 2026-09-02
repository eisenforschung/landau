import warnings
from functools import cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyiron_snippets.import_alarm import ImportAlarm

from landau.calculate import calc_phase_diagram
from landau.interpolate import WhitneyTemperatureInterpolator
from landau.phases import LinePhase, TemperatureDependentLinePhase

with ImportAlarm() as phonopy_alarm:
    from phonopy import Phonopy
    from phonopy.physical_units import get_physical_units
    from phonopy.structure.atoms import PhonopyAtoms

    from landau.phases.quasiharmonic import (
        DynamicalInstabilityWarning,
        EosExtrapolationWarning,
        PhonopyQuasiHarmonicPhase,
        _eos_curve,
        _lowest_frequency,
    )

pytestmark = pytest.mark.skipif(phonopy_alarm.message is not None, reason="phonopy is not installed")


# --- analytic references -----------------------------------------------------------------
#
# An Einstein solid: every one of the 3N modes sits at the same frequency, so the free
# energy per atom is three times the single-oscillator expression and is independent of
# the mesh.  This is the oracle the mode sum is checked against -- it is derived from the
# statistics, not from phonopy, so agreement means the conventions match.


def einstein_free_energy(omega, T, classical=False):
    """Harmonic free energy per atom of an Einstein solid at ``omega`` THz, in eV."""
    units = get_physical_units()
    hw = units.THzToEv * omega
    kB = units.KB
    T = np.asarray(T, dtype=float)
    hot = T > 0
    safe = np.where(hot, T, 1.0)
    if classical:
        # no T -> 0 limit; phonopy reports zero there
        return np.where(hot, 3 * kB * safe * np.log(hw / (kB * safe)), 0.0)
    x = hw / (kB * safe)
    return np.where(hot, 3 * (0.5 * hw + kB * safe * np.log1p(-np.exp(-x))), 1.5 * hw)


# --- building an Einstein solid through phonopy ------------------------------------------
#
# Uncoupled, isotropic force constants (Phi_ii = c * I, Phi_ij = 0) make the dynamical
# matrix c/m * I at every q, so every branch is flat at sqrt(c/m).  That is an Einstein
# solid built through phonopy's own machinery: a real Phonopy object, a real mesh and a
# real run_thermal_properties, planted so the answer is known in closed form.

CELLS = {
    # tag: symbols, scaled positions, primitive_matrix.  An explicit identity keeps
    # phonopy 3 and 4 on the same primitive cell for the simple-cubic case; "auto" reduces
    # the conventional fcc cell by four, which is the case worth exercising.
    "sc": (["Cu"], [[0, 0, 0]], np.eye(3)),
    "fcc": (["Cu"] * 4, [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], "auto"),
}


@cache
def einstein_phonopy(omega, volume=20.0, classical=False, cell="sc", unstable_direction=None, mesh=(4, 4, 4)):
    """A ``Phonopy`` whose every mode sits at ``omega`` THz, thermal properties already run.

    ``unstable_direction`` flips the sign of one Cartesian force constant, which phonopy
    reports as a branch at ``-omega``: an imaginary mode, exactly what a dynamically
    unstable volume looks like.

    Cached because building one is the dominant cost of this file and the result is a pure
    function of the arguments.  Callers only ever read the frequencies and reset the
    temperatures before running, so sharing an object between phases is safe -- except
    where the point of the test is that two phases were built separately, which reaches
    past the cache through ``einstein_phonopy.__wrapped__``.
    """
    symbols, positions, primitive_matrix = CELLS[cell]
    lattice_constant = (volume * len(symbols)) ** (1 / 3)
    atoms = PhonopyAtoms(
        symbols=list(symbols),
        cell=np.eye(3) * lattice_constant,
        scaled_positions=list(positions),
    )
    phonon = Phonopy(atoms, supercell_matrix=np.eye(3) * 2, primitive_matrix=primitive_matrix)
    n = len(phonon.supercell)
    stiffness = phonon.supercell.masses[0] * (omega / get_physical_units().DefaultToTHz) ** 2
    force_constants = np.zeros((n, n, 3, 3))
    for i in range(n):
        force_constants[i, i] = stiffness * np.eye(3)
        if unstable_direction is not None:
            force_constants[i, i, unstable_direction, unstable_direction] *= -1
    phonon.force_constants = force_constants
    phonon.run_mesh(list(mesh), is_gamma_center=True)
    # the temperature here is arbitrary: the phase resets it on every evaluation.  What
    # the caller does fix by running this is the mesh, the statistics and the cutoff.
    phonon.run_thermal_properties(temperatures=[0.0], classical=classical)
    return phonon


def einstein_phase(name="einstein", omegas=5.0, volumes=None, energies=None, classical=False, fresh=False, **kwargs):
    """A phase of Einstein solids, one per volume, through the primary constructor.

    ``omegas`` may be a scalar (the same frequency at every volume, so no thermal
    expansion) or one frequency per volume.  ``fresh`` bypasses the phonopy cache so the
    phase holds objects nothing else shares.
    """
    if volumes is None:
        volumes = np.linspace(17.0, 23.0, 7)
    volumes = np.asarray(volumes, dtype=float)
    omegas = np.broadcast_to(np.asarray(omegas, dtype=float), volumes.shape)
    if energies is None:
        energies = [vinet_energy(v) for v in volumes]
    build = einstein_phonopy.__wrapped__ if fresh else einstein_phonopy
    phonopys = [build(w, volume=v, classical=classical) for v, w in zip(volumes, omegas, strict=True)]
    return PhonopyQuasiHarmonicPhase(
        name,
        0.0,
        thermal_properties=[p.thermal_properties for p in phonopys],
        volumes=volumes,
        energies=energies,
        atoms_per_cell=1,
        atoms_per_primitive_cell=1,
        **kwargs,
    )


VINET = {"E0": -3.5, "B0": 0.6, "B0p": 4.2, "V0": 20.0}


def vinet_energy(volume):
    """Static energy per atom exactly on a Vinet curve with the parameters above.

    Written out rather than taken from :func:`phonopy.qha.eos.get_eos`, both because the
    planted-minimum test is then an independent check of phonopy's fit and because the
    signature of those closures changed between phonopy 3 and 4.
    """
    E0, B0, B0p, V0 = (VINET[k] for k in ("E0", "B0", "B0p", "V0"))
    x = np.cbrt(np.asarray(volume, dtype=float) / V0)
    xi = 1.5 * (B0p - 1)
    return float(E0 + 9 * B0 * V0 / xi**2 * (1 + (xi * (1 - x) - 1) * np.exp(xi * (1 - x))))


def planted_phase(name="planted", omega=5.0, volumes=None, **kwargs):
    """A phase whose static energy is exactly Vinet and whose spectrum does not depend on V.

    The vibrational free energy is then the same constant at every volume, so the total
    free energy is that same Vinet curve shifted, and the fit must recover ``V0`` exactly
    and ``E0`` shifted by exactly the vibrational free energy.
    """
    return einstein_phase(name, omegas=omega, volumes=volumes, **kwargs)


def grueneisen_phase(name="grueneisen", omega0=5.0, gamma=2.0, volumes=None, **kwargs):
    """Vinet static energy plus mode frequencies softening as ``(V0 / V) ** gamma``.

    That volume dependence is what makes the solid expand with temperature, so this is
    the phase to use whenever the test is about ``V(T)``.
    """
    if volumes is None:
        volumes = np.linspace(17.0, 23.0, 7)
    volumes = np.asarray(volumes, dtype=float)
    return einstein_phase(name, omegas=omega0 * (VINET["V0"] / volumes) ** gamma, volumes=volumes, **kwargs)


# --- the mode sum ------------------------------------------------------------------------
#
# helmholtz_free_energies returns static + vibrational per atom for every sampled volume; the
# planted phases put the same frequency at every volume, so one entry is the whole story.


def vibrational(phase, T):
    """The vibrational part alone, for a phase whose static energies are all zero."""
    return phase.helmholtz_free_energies(T)


@pytest.mark.parametrize("classical", [False, True])
def test_vibrational_free_energy_matches_the_einstein_closed_form(classical):
    phase = einstein_phase(omegas=6.5, energies=np.zeros(7), classical=classical)
    for T in (0.0, 1.0, 50.0, 300.0, 1200.0, 3000.0):
        assert_allclose(
            vibrational(phase, T),
            einstein_free_energy(6.5, T, classical=classical),
            rtol=1e-12,
            atol=0,
        )


def test_zero_point_energy_is_half_hbar_omega_per_mode():
    phase = einstein_phase(omegas=6.5, energies=np.zeros(7))
    hw = get_physical_units().THzToEv * 6.5
    assert_allclose(vibrational(phase, 0.0), 1.5 * hw, rtol=1e-14)


def test_classical_free_energy_vanishes_at_zero_temperature():
    # the classical expression has no T -> 0 limit; phonopy reports exactly zero
    phase = einstein_phase(omegas=6.5, energies=np.zeros(7), classical=True)
    assert np.all(vibrational(phase, 0.0) == 0.0)


def test_quantum_exceeds_classical_by_the_zero_point_energy_at_low_temperature():
    phase = einstein_phase(omegas=6.5, energies=np.zeros(7))
    hw = get_physical_units().THzToEv * 6.5
    # at 10 K the thermal occupation of a 6.5 THz mode is e^-31, so quantum is pure ZPE
    assert_allclose(vibrational(phase, 10.0), 1.5 * hw, rtol=1e-12)


def test_quantum_minus_classical_follows_the_high_temperature_expansion():
    # F_quantum - F_classical -> 3 (hbar omega)^2 / (24 kB T) as kB T >> hbar omega,
    # a statement about the statistics rather than about this implementation
    omega, T = 2.0, 4000.0
    units = get_physical_units()
    hw = units.THzToEv * omega
    quantum = einstein_phase(omegas=omega, energies=np.zeros(7))
    classical = einstein_phase(omegas=omega, energies=np.zeros(7), classical=True)
    difference = vibrational(quantum, T) - vibrational(classical, T)
    assert_allclose(difference, 3 * hw**2 / (24 * units.KB * T), rtol=1e-3)


def test_free_energy_per_atom_normalises_by_the_primitive_cell_not_the_unit_cell():
    # a conventional fcc cell holds four atoms but reduces to one primitive atom, so the
    # spectrum carries three bands while volume and energy describe four atoms.  Dividing
    # the mode sum by four instead of one is the mistake this pins.
    per_atom_energy, per_atom_volume, omega = -3.25, 5.0, 6.5
    phonopys = [
        einstein_phonopy(omega, volume=v, cell="fcc")
        for v in (4.6, 4.8, 5.0, 5.2, 5.4)
    ]
    assert len(phonopys[0].unitcell) == 4 and len(phonopys[0].primitive) == 1
    phase = PhonopyQuasiHarmonicPhase(
        "fcc",
        0.0,
        thermal_properties=[p.thermal_properties for p in phonopys],
        volumes=[4 * v for v in (4.6, 4.8, 5.0, 5.2, 5.4)],
        energies=[4 * per_atom_energy] * 5,
        atoms_per_cell=4,
        atoms_per_primitive_cell=1,
    )
    assert_allclose(phase.volumes_per_atom, (4.6, 4.8, 5.0, 5.2, 5.4), rtol=1e-14)
    assert_allclose(phase.energies_per_atom, [per_atom_energy] * 5, rtol=1e-14)
    for T in (0.0, 300.0, 1000.0):
        expected = per_atom_energy + einstein_free_energy(omega, T)
        assert_allclose(phase.helmholtz_free_energies(T), expected, rtol=1e-12)
    assert per_atom_volume in phase.volumes_per_atom


def test_free_energy_is_independent_of_how_the_same_crystal_is_celled():
    # the same physics described per primitive cell and per four-atom cell must give the
    # same per-atom numbers
    volumes = np.array([4.6, 4.8, 5.0, 5.2, 5.4])
    one = einstein_phase("one", omegas=6.5, volumes=volumes, energies=[-3.25] * 5)
    four_phonopys = [
        einstein_phonopy(6.5, volume=v, cell="fcc") for v in volumes
    ]
    four = PhonopyQuasiHarmonicPhase(
        "four",
        0.0,
        thermal_properties=[p.thermal_properties for p in four_phonopys],
        volumes=4 * volumes,
        energies=[4 * -3.25] * 5,
        atoms_per_cell=4,
        atoms_per_primitive_cell=1,
    )
    for T in (0.0, 300.0, 1500.0):
        assert_allclose(one.helmholtz_free_energies(T), four.helmholtz_free_energies(T), rtol=1e-14)


def test_the_temperatures_the_caller_ran_with_do_not_constrain_the_phase():
    # the phase resets ThermalProperties.temperatures and re-runs, so whatever grid the
    # caller happened to pass to run_thermal_properties is irrelevant
    phase = einstein_phase(omegas=6.5, energies=np.zeros(7))
    for tp in phase.thermal_properties:
        tp.temperatures = np.array([11.0, 22.0, 33.0])
        tp.run()
    assert_allclose(vibrational(phase, 777.0), einstein_free_energy(6.5, 777.0), rtol=1e-12)


def test_negative_temperatures_are_rejected():
    # phonopy's ThermalProperties.temperatures setter silently drops negative entries,
    # which would return fewer values than were asked for and misalign them
    phase = planted_phase()
    with pytest.raises(ValueError, match="non-negative"):
        phase.helmholtz_free_energies(-10.0)


def test_arbitrary_unsorted_temperatures_are_answered_in_order():
    phase = planted_phase()
    T = np.array([900.0, 17.0, 400.0, 0.0])
    assert_allclose(phase.line_free_energy(T), [phase.line_free_energy(t) for t in T], rtol=1e-14)


# --- the volume minimisation -------------------------------------------------------------


def test_planted_vinet_minimum_is_recovered_exactly():
    # frequencies do not depend on volume here, so the total free energy is the planted
    # Vinet curve plus a constant: the fit must return V0 unchanged and E0 shifted by
    # exactly the vibrational free energy
    phase = planted_phase(omega=5.0)
    for T in (0.0, 300.0, 1500.0):
        energy, bulk_modulus, bulk_modulus_prime, volume = phase.eos_parameters(T)
        assert_allclose(volume, VINET["V0"], rtol=1e-8)
        assert_allclose(bulk_modulus, VINET["B0"], rtol=1e-6)
        assert_allclose(bulk_modulus_prime, VINET["B0p"], rtol=1e-6)
        assert_allclose(energy, VINET["E0"] + einstein_free_energy(5.0, np.asarray(T)), rtol=1e-8)


def test_volume_independent_spectrum_does_not_expand():
    phase = planted_phase()
    volumes = phase.equilibrium_volume(np.array([0.0, 500.0, 1500.0]))
    assert_allclose(volumes, VINET["V0"], rtol=1e-8)


def test_softening_modes_produce_monotonic_thermal_expansion():
    phase = grueneisen_phase(gamma=2.0)
    T = np.array([0.0, 200.0, 400.0, 800.0, 1200.0])
    volumes = phase.equilibrium_volume(T)
    assert np.all(np.diff(volumes) > 0)
    assert volumes[0] > VINET["V0"]  # zero-point expansion


def test_classical_statistics_remove_the_zero_point_expansion():
    quantum = grueneisen_phase("q", gamma=2.0)
    classical = grueneisen_phase("c", gamma=2.0, classical=True)
    assert_allclose(classical.equilibrium_volume(0.0), VINET["V0"], rtol=1e-8)
    assert quantum.equilibrium_volume(0.0) > classical.equilibrium_volume(0.0)
    assert_allclose(classical.line_free_energy(0.0), VINET["E0"], rtol=1e-8)


def test_free_energy_decreases_with_temperature():
    phase = grueneisen_phase()
    T = np.linspace(0.0, 1200.0, 13)
    assert np.all(np.diff(phase.line_free_energy(T)) < 0)


def test_scalar_temperature_gives_a_scalar_line_free_energy():
    phase = planted_phase()
    out = phase.line_free_energy(300.0)
    assert np.isscalar(out) and not isinstance(out, np.ndarray)
    assert phase.line_free_energy(np.array(300.0)) == out
    assert phase.line_free_energy(np.array([300.0, 500.0])).shape == (2,)


def test_line_phase_interface_is_satisfied():
    phase = planted_phase()
    assert phase.line_concentration == 0.0
    assert phase.free_energy(300.0, 0.0) == phase.line_free_energy(300.0)
    assert_allclose(phase.semigrand_potential(300.0, 0.7), phase.line_free_energy(300.0))
    assert phase.concentration(300.0, 0.7) == 0.0


# --- agreement with the fitted line phase ------------------------------------------------


def test_matches_a_temperature_dependent_line_phase_exactly_on_its_own_samples():
    # Whitney is the interpolating TemperatureInterpolator landau ships (smoothing 0), so it
    # reproduces its input at the samples by construction; between them the two only agree to
    # the quality of the fit, and that gap has to close as the grid is refined
    phase = grueneisen_phase()
    off_node = np.linspace(150.0, 1050.0, 11)
    direct = phase.line_free_energy(off_node)
    errors = []
    for n in (6, 11, 21):
        T = np.linspace(100.0, 1100.0, n)
        fitted = TemperatureDependentLinePhase(
            "fitted",
            fixed_concentration=0.0,
            temperatures=T,
            free_energies=phase.line_free_energy(T),
            interpolator=WhitneyTemperatureInterpolator(),
        )
        assert_allclose(fitted.line_free_energy(T), phase.line_free_energy(T), rtol=1e-10)
        errors.append(float(np.abs(np.ravel(fitted.line_free_energy(off_node)) - direct).max()))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[-1] > 10  # genuinely converging, not a constant offset
    assert errors[-1] < 1e-5


# --- dynamical instability ---------------------------------------------------------------


def unstable_pieces(volumes, unstable, omega=5.0):
    """``(thermal_properties, mesh minima)`` with an imaginary branch where asked.

    The minima come off the mesh rather than out of the phase, so they are an independent
    statement of which volumes ought to be refused.
    """
    phonopys = [
        einstein_phonopy(omega, volume=v, unstable_direction=0 if i in unstable else None)
        for i, v in enumerate(volumes)
    ]
    return (
        [p.thermal_properties for p in phonopys],
        [float(p.mesh.frequencies.min()) for p in phonopys],
    )


@pytest.mark.parametrize("unstable_direction,sign", [(None, 1), (0, -1)])
def test_lowest_frequency_matches_the_mesh(unstable_direction, sign):
    # the screen reads ThermalProperties._frequencies, which phonopy does not expose.  This
    # is the test that fails if that attribute moves, and the phonopy upper bound is pinned
    # to the minor release so a new one arrives as a dependabot PR that runs it
    phonon = einstein_phonopy(5.0, volume=20.0, unstable_direction=unstable_direction)
    assert _lowest_frequency(phonon.thermal_properties) == pytest.approx(sign * 5.0, abs=1e-10)
    assert _lowest_frequency(phonon.thermal_properties) == pytest.approx(
        float(phonon.mesh.frequencies.min()), abs=1e-10
    )


def test_unstable_volumes_are_dropped_with_a_warning():
    volumes = np.linspace(17.0, 23.0, 7)
    thermal_properties, lowest = unstable_pieces(volumes, {0})
    assert lowest[0] < 0 and lowest[1] > 0
    with pytest.warns(DynamicalInstabilityWarning, match="dropped 1 of 7"):
        phase = PhonopyQuasiHarmonicPhase(
            "unstable", 0.0,
            thermal_properties=thermal_properties,
            volumes=volumes, energies=[vinet_energy(v) for v in volumes],
            atoms_per_cell=1, atoms_per_primitive_cell=1,
        )
    assert phase.lowest_frequencies == pytest.approx(lowest, abs=1e-10)
    assert len(phase.sampled_volumes) == 6
    assert volumes[0] not in phase.sampled_volumes


def test_keeping_an_unstable_volume_changes_the_answer():
    # imaginary modes are silently skipped by the mode sum, so the bad volume still
    # returns a smooth plausible number -- and drags the fit with it
    volumes = np.linspace(17.0, 23.0, 7)
    thermal_properties, _ = unstable_pieces(volumes, {0})
    pieces = {
        "thermal_properties": thermal_properties,
        "volumes": volumes,
        "energies": [vinet_energy(v) for v in volumes],
        "atoms_per_cell": 1,
        "atoms_per_primitive_cell": 1,
    }
    with pytest.warns(DynamicalInstabilityWarning):
        dropped = PhonopyQuasiHarmonicPhase("dropped", 0.0, **pieces)
    kept = PhonopyQuasiHarmonicPhase("kept", 0.0, **pieces, min_frequency=-10.0)
    assert len(kept.sampled_volumes) == 7
    assert abs(kept.line_free_energy(600.0) - dropped.line_free_energy(600.0)) > 1e-4


def test_too_few_stable_volumes_is_an_error_not_a_fit():
    volumes = np.linspace(17.0, 23.0, 5)
    thermal_properties, _ = unstable_pieces(volumes, {0, 1})
    with pytest.raises(ValueError, match="needs at least four"):
        PhonopyQuasiHarmonicPhase(
            "mostly unstable", 0.0,
            thermal_properties=thermal_properties,
            volumes=volumes, energies=[vinet_energy(v) for v in volumes],
            atoms_per_cell=1, atoms_per_primitive_cell=1,
        )


# --- the extrapolation ceiling -----------------------------------------------------------


def test_past_the_sampled_volumes_the_volume_is_clamped_and_warned_about():
    # a narrow volume window that thermal expansion runs out of.  Past the ceiling the
    # minimisation is constrained to the sampled volumes rather than letting the equation
    # of state extrapolate to reach the true minimum
    phase = grueneisen_phase(gamma=3.0, volumes=np.linspace(19.8, 20.6, 5))
    ceiling = phase.max_temperature(upper=4000.0, tolerance=0.5)
    assert 0 < ceiling < 4000.0

    inside = phase.line_free_energy(ceiling - 1.0)
    assert np.isfinite(inside)
    with pytest.warns(EosExtrapolationWarning, match="outside the sampled range"):
        outside = phase.line_free_energy(ceiling + 1.0)
    with pytest.warns(EosExtrapolationWarning):
        volume = phase.equilibrium_volume(ceiling + 1.0)
    # the volume stops at the edge of the data instead of following the fit past it
    assert volume == phase.sampled_volumes[-1]
    assert np.isfinite(outside) and outside < inside


def test_the_clamped_branch_joins_the_free_one_continuously():
    # Comparing the two branches at the same temperature, not the curve either side of the
    # crossing -- across any interval the curve also moves by dF/dT, which swamps this.
    # The constrained minimum can never beat the free one, and the gap between them is
    # second order in how far the free minimum has overshot the edge, so it vanishes at
    # the crossing: no step for calc_phase_diagram to mistake for a transition.
    phase = grueneisen_phase(gamma=3.0, volumes=np.linspace(19.8, 20.6, 5))
    ceiling = phase.max_temperature(upper=4000.0, tolerance=1e-3)

    def gap(T):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EosExtrapolationWarning)
            return phase.line_free_energy(T) - phase.eos_parameters(T)[0]

    assert 0 <= gap(ceiling + 1e-2) < 1e-10
    # and far enough past it the constraint does bite, so this is not a no-op
    assert gap(ceiling + 200.0) > 1e-4


def test_no_warning_while_the_equilibrium_volume_is_still_inside():
    phase = grueneisen_phase(gamma=3.0, volumes=np.linspace(19.8, 20.6, 5))
    ceiling = phase.max_temperature(upper=4000.0, tolerance=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EosExtrapolationWarning)
        phase.line_free_energy(np.linspace(0.0, ceiling - 1.0, 9))


def test_a_wide_enough_volume_range_has_no_ceiling_below_the_bracket():
    phase = grueneisen_phase(gamma=2.0, volumes=np.linspace(17.0, 26.0, 9))
    assert phase.max_temperature(upper=1500.0) == 1500.0


def test_the_phase_stays_in_the_diagram_above_its_ceiling():
    # extrapolating is a warning, not a withdrawal: calc_phase_diagram still gets a number
    # for the phase above its ceiling and can still pick it
    solid = grueneisen_phase("solid", gamma=3.0, volumes=np.linspace(19.8, 20.6, 5))
    ceiling = solid.max_temperature(upper=4000.0, tolerance=0.5)
    other = LinePhase("other", 0.0, line_energy=0.0, line_entropy=0.0)
    T = np.array([ceiling - 50.0, ceiling + 50.0])
    with pytest.warns(EosExtrapolationWarning):
        df = calc_phase_diagram([solid, other], Ts=T, mu=np.array([0.0]), refine=False)
    above = df.query("T > @ceiling")
    assert above.phase.tolist() == ["solid"]
    assert np.isfinite(above.phi).all()


# --- the equation-of-state curve and the debug plot -------------------------------------


def test_eos_curve_reproduces_the_planted_vinet():
    # pins both the parameter order out of fit_to_eos and phonopy's calling convention,
    # which differs between the two majors: 4 takes eos(v, p) and 3 takes eos(v, *p)
    volumes = np.linspace(17.0, 23.0, 7)
    parameters = (VINET["E0"], VINET["B0"], VINET["B0p"], VINET["V0"])
    assert_allclose(
        _eos_curve("vinet", volumes, parameters),
        [vinet_energy(v) for v in volumes],
        rtol=1e-12,
    )


def test_check_equation_of_state_draws_the_samples_the_fit_and_the_minimum():
    matplotlib.use("Agg")
    phase = grueneisen_phase(gamma=2.0)
    T = 600.0
    plt.figure()
    try:
        phase.check_equation_of_state(T, samples=501)
        ax = plt.gca()
        (curve,) = ax.lines
        samples, minimum = ax.collections[0], ax.collections[1]

        assert curve.get_xydata().shape == (501, 2)
        assert_allclose(samples.get_offsets()[:, 0], phase.sampled_volumes, rtol=1e-12)
        assert_allclose(samples.get_offsets()[:, 1], phase.helmholtz_free_energies(T), rtol=1e-12)
        # the marker is the number the phase actually reports, and the drawn curve dips to it
        assert_allclose(minimum.get_offsets()[0], [phase.equilibrium_volume(T), phase.line_free_energy(T)], rtol=1e-12)
        # the reported minimum is the true minimum of the drawn function, so the sampled
        # curve may sit above it by the grid resolution but never below it
        drawn = curve.get_xydata()[:, 1].min()
        assert 0 <= drawn - phase.line_free_energy(T) < 1e-6
    finally:
        plt.close("all")


# --- construction contract and identity --------------------------------------------------


def test_parallel_sequences_must_have_the_same_length():
    phase = planted_phase()
    with pytest.raises(ValueError, match="parallel sequences"):
        PhonopyQuasiHarmonicPhase(
            "ragged", 0.0,
            thermal_properties=phase.thermal_properties,
            volumes=phase.volumes[:-1], energies=phase.energies,
            atoms_per_cell=1, atoms_per_primitive_cell=1,
        )


def test_duplicate_volumes_are_rejected():
    with pytest.raises(ValueError, match="must be distinct"):
        einstein_phase("duplicate", volumes=[17.0, 19.0, 21.0, 23.0, 19.0])


def test_non_positive_volume_is_rejected():
    phase = planted_phase()
    with pytest.raises(ValueError, match="volumes must be positive"):
        PhonopyQuasiHarmonicPhase(
            "flat", 0.0,
            thermal_properties=phase.thermal_properties,
            volumes=[0.0] + list(phase.volumes[1:]), energies=phase.energies,
            atoms_per_cell=1, atoms_per_primitive_cell=1,
        )


@pytest.mark.parametrize("count", ["atoms_per_cell", "atoms_per_primitive_cell"])
def test_atom_counts_must_be_positive(count):
    phase = planted_phase()
    counts = {"atoms_per_cell": 1, "atoms_per_primitive_cell": 1, count: 0}
    with pytest.raises(ValueError, match=f"{count} must be positive"):
        PhonopyQuasiHarmonicPhase(
            "zero", 0.0,
            thermal_properties=phase.thermal_properties,
            volumes=phase.volumes, energies=phase.energies,
            **counts,
        )


def test_unknown_equation_of_state_is_rejected():
    with pytest.raises(ValueError, match="eos must be one of"):
        planted_phase(eos="debye")


def test_phases_hash_and_compare_by_content():
    # a ThermalProperties compares by identity, so the phase pickles them, as AsePhase
    # does for its ThermoChem: two phases built from equivalent inputs must agree
    a = planted_phase("p")
    b = planted_phase("p", fresh=True)
    assert a.thermal_properties[0] is not b.thermal_properties[0]
    assert a == b and hash(a) == hash(b)
    assert {a, b} == {a}
    assert a != planted_phase("other")
    assert a != planted_phase("p", eos="murnaghan")
    assert a != planted_phase("p", omega=5.5)
    assert a != einstein_phase("p", omegas=5.0, classical=True)


def test_equality_survives_being_evaluated():
    # a ThermalProperties pickle carries the temperature it was last run at, and this class
    # resets that on every evaluation, so a key read live would drift as the phase is used
    a = planted_phase("p")
    b = planted_phase("p", fresh=True)
    a.line_free_energy(1234.0)
    assert a == b and hash(a) == hash(b)


def test_equation_of_state_choice_changes_the_answer():
    vinet = grueneisen_phase("v")
    murnaghan = grueneisen_phase("m", eos="murnaghan")
    assert vinet.line_free_energy(900.0) != murnaghan.line_free_energy(900.0)


# --- the live Phonopy path ----------------------------------------------------------------


def test_from_phonopy_derives_the_volume_and_both_atom_counts():
    volumes = np.array([4.6, 4.8, 5.0, 5.2, 5.4])
    phonopys = [
        einstein_phonopy(5.0, volume=v, cell="fcc") for v in volumes
    ]
    energies = [4 * vinet_energy(4 * v) for v in volumes]
    phase = PhonopyQuasiHarmonicPhase.from_phonopy("fcc", 0.0, phonopys, energies)
    assert phase.atoms_per_cell == 4 and phase.atoms_per_primitive_cell == 1
    assert_allclose(phase.volumes, [p.unitcell.volume for p in phonopys], rtol=1e-14)
    assert_allclose(phase.volumes_per_atom, volumes, rtol=1e-12)
    # the screen comes for free on this path, off the mesh phonopy already holds
    assert_allclose(phase.lowest_frequencies, 5.0, rtol=1e-10)


def test_from_phonopy_requires_thermal_properties_to_have_been_run():
    # mesh, statistics and mode cutoff are all chosen in run_thermal_properties, so the
    # caller runs it; a default chosen here would silently override their intent
    phonopys = [einstein_phonopy(5.0, volume=v) for v in (17.0, 19.0, 21.0, 23.0)]
    phonopys[2] = Phonopy(
        PhonopyAtoms(symbols=["Cu"], cell=np.eye(3) * 21.0 ** (1 / 3), scaled_positions=[[0, 0, 0]]),
        supercell_matrix=np.eye(3) * 2,
        primitive_matrix=np.eye(3),
    )
    assert phonopys[2].thermal_properties is None
    with pytest.raises(ValueError, match=r"positions \[2\] carry no thermal properties"):
        PhonopyQuasiHarmonicPhase.from_phonopy("Cu", 0.0, phonopys, [0.0] * 4)


def test_from_phonopy_rejects_a_mixed_set_of_structures():
    phonopys = [einstein_phonopy(5.0, volume=v) for v in (17.0, 19.0, 21.0, 23.0)]
    phonopys[1] = einstein_phonopy(5.0, volume=19.0, cell="fcc")
    with pytest.raises(ValueError, match="same structure"):
        PhonopyQuasiHarmonicPhase.from_phonopy("Cu", 0.0, phonopys, [0.0] * 4)


@pytest.mark.parametrize("classical", [False, True])
def test_from_phonopy_reproduces_phonopys_own_thermal_properties(classical):
    # the phase must return phonopy's number, not a second opinion on it
    phonopys = [morse_crystal(a, classical=classical) for a in np.linspace(3.52, 3.90, 6)]
    phase = PhonopyQuasiHarmonicPhase.from_phonopy(
        "Cu", 0.0, [p for p, _ in phonopys], [e for _, e in phonopys]
    )
    T = 300.0
    phonon, energy = phonopys[0]
    phonon.run_thermal_properties(temperatures=[T], classical=classical)
    reference = phonon.thermal_properties.thermal_properties[1][0] / get_physical_units().EvTokJmol
    assert_allclose(phase.helmholtz_free_energies(T)[0], energy / 4 + reference, rtol=1e-14)


def test_from_phonopy_expands_when_heated():
    # Morse force constants soften as the crystal expands, so it expands when heated.  The
    # temperature comes from the phase's own ceiling rather than a guess, since how far this
    # toy potential expands before it runs out of sampled volumes is not something to hard-code
    phonopys = [morse_crystal(a) for a in np.linspace(3.52, 3.90, 6)]
    phase = PhonopyQuasiHarmonicPhase.from_phonopy(
        "Cu", 0.0, [p for p, _ in phonopys], [e for _, e in phonopys]
    )
    assert len(phase.sampled_volumes) == 6
    ceiling = phase.max_temperature(upper=2000.0)
    assert ceiling > 0
    warm = 0.5 * ceiling
    assert phase.line_free_energy(warm) < phase.line_free_energy(0.0)
    assert phase.equilibrium_volume(warm) > phase.equilibrium_volume(0.0)


# A real phonopy calculation, with forces from a nearest-neighbour Morse pair potential rather
# than a calculator, so this exercises `from_phonopy` end to end without pulling in a second
# optional dependency.  Morse rather than a harmonic spring because harmonic force constants do
# not depend on volume, which would leave the crystal with no thermal expansion to check.

MORSE = {"depth": 0.3, "alpha": 1.5, "distance": 2.55, "cutoff": 3.0}


def morse_forces_and_energy(cell, positions):
    """Forces in eV/A and total energy in eV of a nearest-neighbour Morse crystal."""
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    # delta[i, j] is the minimum-image vector from atom i to atom j; the cells used below are
    # cubic, so rounding the fractional offsets is the exact minimum image
    delta = positions[None, :, :] - positions[:, None, :]
    fractional = delta @ np.linalg.inv(cell)
    delta = (fractional - np.round(fractional)) @ cell
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)

    inside = distance < MORSE["cutoff"]
    safe = np.where(inside, distance, MORSE["distance"])
    decay = np.exp(-MORSE["alpha"] * (safe - MORSE["distance"]))
    energy = 0.5 * np.where(inside, MORSE["depth"] * (1 - decay) ** 2, 0.0).sum()
    gradient = np.where(inside, 2 * MORSE["depth"] * MORSE["alpha"] * (1 - decay) * decay, 0.0)
    return ((gradient / safe)[..., None] * delta).sum(axis=1), energy


def morse_crystal(a, classical=False, supercell=2, mesh=(8, 8, 8)):
    """``(Phonopy, energy)`` for a conventional fcc cell of lattice constant ``a``.

    The cell holds four atoms and reduces to one primitive atom, so this is also the case
    where the spectrum and the energy are normalised by different atom counts.
    """
    symbols, positions, _ = CELLS["fcc"]
    atoms = PhonopyAtoms(symbols=list(symbols), cell=np.eye(3) * a, scaled_positions=list(positions))
    _, energy = morse_forces_and_energy(atoms.cell, atoms.positions)
    phonon = Phonopy(atoms, supercell_matrix=np.eye(3) * supercell, primitive_matrix="auto")
    phonon.generate_displacements(distance=0.01)
    phonon.forces = np.array(
        [morse_forces_and_energy(sc.cell, sc.positions)[0] for sc in phonon.supercells_with_displacements]
    )
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.run_mesh(list(mesh), is_gamma_center=True)
    phonon.run_thermal_properties(temperatures=[0.0], classical=classical)
    return phonon, energy


# --- integration with calc_phase_diagram --------------------------------------------------


def test_calc_phase_diagram_locates_the_transition_the_free_energies_imply():
    solid = grueneisen_phase("solid", gamma=2.0)
    # a competing phase built to cross the solid at exactly this temperature, with an extra
    # 0.8 kB/atom of entropy so the crossing is a clean single sign change
    expected = 700.0
    entropy = -float(np.gradient(solid.line_free_energy(np.array([expected - 1, expected, expected + 1])))[1])
    melt_entropy = entropy + 0.8 * get_physical_units().KB
    melt = LinePhase(
        "melt",
        0.0,
        line_energy=float(solid.line_free_energy(expected)) + expected * melt_entropy,
        line_entropy=melt_entropy,
    )
    assert_allclose(melt.line_free_energy(expected), solid.line_free_energy(expected), rtol=1e-12)

    df = calc_phase_diagram([solid, melt], Ts=np.linspace(300.0, 1100.0, 17), mu=np.array([0.0]))
    df = df.sort_values("T")
    phases = df["phase"].to_numpy()
    # refinement can emit several rows at the same temperature, so compare distinct ones
    switches = np.unique(df["T"].to_numpy()[:-1][phases[:-1] != phases[1:]])
    assert len(switches) == 1
    assert abs(switches[0] - expected) < 1.0
