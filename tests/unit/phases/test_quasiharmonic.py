import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyiron_snippets.import_alarm import ImportAlarm

from landau.calculate import calc_phase_diagram
from landau.interpolate import WhitneyTemperatureInterpolator
from landau.phases import LinePhase, TemperatureDependentLinePhase

with ImportAlarm() as phonopy_alarm:
    from phonopy.physical_units import get_physical_units

    from landau.phases.quasiharmonic import (
        DynamicalInstabilityWarning,
        EosExtrapolationWarning,
        PhononSpectrum,
        PhonopyQuasiHarmonicPhase,
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


def einstein_spectrum(
    omega,
    volume=20.0,
    energy=-3.0,
    atoms_per_cell=1,
    atoms_per_primitive_cell=1,
    n_qpoints=5,
):
    """A spectrum whose every mode sits at ``omega`` THz, with uneven q-point weights."""
    frequencies = np.full((n_qpoints, 3 * atoms_per_primitive_cell), float(omega))
    weights = np.arange(1, n_qpoints + 1)
    return PhononSpectrum(
        frequencies=frequencies,
        weights=weights,
        volume=volume,
        energy=energy,
        atoms_per_cell=atoms_per_cell,
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
    if volumes is None:
        volumes = np.linspace(17.0, 23.0, 7)
    spectra = tuple(einstein_spectrum(omega, volume=v, energy=vinet_energy(v)) for v in volumes)
    return PhonopyQuasiHarmonicPhase(name, 0.0, spectra=spectra, **kwargs)


def grueneisen_phase(name="grueneisen", omega0=5.0, gamma=2.0, volumes=None, **kwargs):
    """Vinet static energy plus mode frequencies softening as ``(V0 / V) ** gamma``.

    That volume dependence is what makes the solid expand with temperature, so this is
    the phase to use whenever the test is about ``V(T)``.
    """
    if volumes is None:
        volumes = np.linspace(17.0, 23.0, 7)
    spectra = tuple(
        einstein_spectrum(omega0 * (VINET["V0"] / v) ** gamma, volume=v, energy=vinet_energy(v)) for v in volumes
    )
    return PhonopyQuasiHarmonicPhase(name, 0.0, spectra=spectra, **kwargs)


# --- PhononSpectrum: construction contract -----------------------------------------------


def test_atoms_per_primitive_cell_is_a_third_of_the_band_count():
    spec = einstein_spectrum(4.0, atoms_per_primitive_cell=3, atoms_per_cell=12)
    assert spec.frequencies.shape[1] == 9
    assert spec.atoms_per_primitive_cell == 3
    assert spec.atoms_per_cell == 12


def test_band_count_must_be_a_multiple_of_three():
    with pytest.raises(ValueError, match="multiple of three"):
        PhononSpectrum(frequencies=np.ones((4, 5)), weights=np.ones(4), volume=1.0, energy=0.0, atoms_per_cell=1)


def test_weights_must_have_one_entry_per_qpoint():
    with pytest.raises(ValueError, match="weights shape"):
        PhononSpectrum(frequencies=np.ones((4, 3)), weights=np.ones(3), volume=1.0, energy=0.0, atoms_per_cell=1)


def test_fractional_weights_are_rejected():
    with pytest.raises(ValueError, match="whole numbers"):
        PhononSpectrum(
            frequencies=np.ones((2, 3)), weights=[0.5, 1.5], volume=1.0, energy=0.0, atoms_per_cell=1
        )


def test_non_positive_volume_is_rejected():
    with pytest.raises(ValueError, match="volume must be positive"):
        PhononSpectrum(frequencies=np.ones((2, 3)), weights=[1, 1], volume=0.0, energy=0.0, atoms_per_cell=1)


def test_frequencies_are_read_only():
    spec = einstein_spectrum(4.0)
    with pytest.raises(ValueError):
        spec.frequencies[0, 0] = 1.0


def test_negative_temperatures_are_rejected():
    # phonopy's ThermalProperties.temperatures setter silently drops negative entries,
    # which would return fewer values than were asked for and misalign them
    spec = einstein_spectrum(4.0)
    with pytest.raises(ValueError, match="non-negative"):
        spec.vibrational_free_energy([-10.0, 0.0, 300.0])


# --- PhononSpectrum: the mode sum --------------------------------------------------------


@pytest.mark.parametrize("classical", [False, True])
def test_vibrational_free_energy_matches_the_einstein_closed_form(classical):
    spec = einstein_spectrum(6.5)
    T = np.array([0.0, 1.0, 50.0, 300.0, 1200.0, 3000.0])
    assert_allclose(
        spec.vibrational_free_energy(T, classical=classical),
        einstein_free_energy(6.5, T, classical=classical),
        rtol=1e-12,
        atol=0,
    )


def test_zero_point_energy_is_half_hbar_omega_per_mode():
    spec = einstein_spectrum(6.5)
    hw = get_physical_units().THzToEv * 6.5
    assert_allclose(spec.vibrational_free_energy(0.0), 1.5 * hw, rtol=1e-14)


def test_classical_free_energy_vanishes_at_zero_temperature():
    # the classical expression has no T -> 0 limit; phonopy reports exactly zero
    assert einstein_spectrum(6.5).vibrational_free_energy(0.0, classical=True) == 0.0


def test_quantum_exceeds_classical_by_the_zero_point_energy_at_low_temperature():
    spec = einstein_spectrum(6.5)
    hw = get_physical_units().THzToEv * 6.5
    # at 10 K the thermal occupation of a 6.5 THz mode is e^-31, so quantum is pure ZPE
    assert_allclose(spec.vibrational_free_energy(10.0), 1.5 * hw, rtol=1e-12)


def test_quantum_minus_classical_follows_the_high_temperature_expansion():
    # F_quantum - F_classical -> 3 (hbar omega)^2 / (24 kB T) as kB T >> hbar omega,
    # a statement about the statistics rather than about this implementation
    omega, T = 2.0, 4000.0
    units = get_physical_units()
    hw = units.THzToEv * omega
    spec = einstein_spectrum(omega)
    difference = spec.vibrational_free_energy(T) - spec.vibrational_free_energy(T, classical=True)
    assert_allclose(difference, 3 * hw**2 / (24 * units.KB * T), rtol=1e-3)


def test_free_energy_per_atom_normalises_by_the_primitive_cell_not_the_unit_cell():
    # a conventional fcc cell holds four atoms but reduces to one primitive atom, so the
    # spectrum carries three bands while volume and energy describe four atoms.  Dividing
    # the mode sum by four instead of one is the mistake this pins.
    per_atom_energy, per_atom_volume, omega = -3.25, 5.0, 6.5
    spec = einstein_spectrum(
        omega,
        volume=4 * per_atom_volume,
        energy=4 * per_atom_energy,
        atoms_per_cell=4,
        atoms_per_primitive_cell=1,
    )
    assert spec.atoms_per_primitive_cell == 1
    assert spec.volume_per_atom == per_atom_volume
    assert spec.static_energy_per_atom == per_atom_energy
    T = np.array([0.0, 300.0, 1000.0])
    assert_allclose(spec.vibrational_free_energy(T), einstein_free_energy(omega, T), rtol=1e-12)
    assert_allclose(spec.free_energy(T), per_atom_energy + einstein_free_energy(omega, T), rtol=1e-12)


def test_free_energy_is_independent_of_how_the_same_crystal_is_celled():
    # the same physics described per primitive cell and per four-atom cell must give the
    # same per-atom numbers
    one = einstein_spectrum(6.5, volume=5.0, energy=-3.25, atoms_per_cell=1, atoms_per_primitive_cell=1)
    four = einstein_spectrum(6.5, volume=20.0, energy=-13.0, atoms_per_cell=4, atoms_per_primitive_cell=1)
    T = np.array([0.0, 300.0, 1500.0])
    assert_allclose(one.free_energy(T), four.free_energy(T), rtol=1e-14)


def test_cutoff_frequency_removes_the_gamma_point_noise():
    # the acoustic branches are exactly zero at Gamma by the sum rule but come back as
    # numerical noise of either sign; the positive ones are kept by phonopy's default
    # cutoff of zero and contribute a spurious kB T ln(hbar omega / kB T)
    noise = np.array([-1.92e-07, 8.29e-08, 1.24e-07])
    frequencies = np.full((3, 3), 6.5)
    frequencies[0] = noise
    weights = np.array([1, 100, 100])
    spec = PhononSpectrum(frequencies=frequencies, weights=weights, volume=20.0, energy=-3.0, atoms_per_cell=1)

    units = get_physical_units()
    T = 2400.0
    kept = noise[noise > 0]
    spurious = (units.KB * T * np.log(units.THzToEv * kept / (units.KB * T))).sum() / weights.sum()

    kept_in = spec.vibrational_free_energy(T, classical=True, cutoff_frequency=0.0)
    cut_out = spec.vibrational_free_energy(T, classical=True, cutoff_frequency=1e-3)
    assert_allclose(kept_in - cut_out, spurious, rtol=1e-12)
    assert spurious < -1e-4  # the artifact is of meV/atom size, not noise


def test_thermal_properties_returns_free_energy_entropy_and_heat_capacity():
    spec = einstein_spectrum(6.5)
    T = np.array([300.0, 900.0])
    free_energy, entropy, heat_capacity = spec.thermal_properties(T)
    assert_allclose(free_energy, spec.vibrational_free_energy(T), rtol=1e-14)
    # the Einstein heat capacity saturates at 3 kB per atom from below
    assert np.all(heat_capacity < 3 * get_physical_units().KB)
    assert np.all(np.diff(heat_capacity) > 0)
    # and S = -dF/dT
    dT = 1e-3
    numerical = -(spec.vibrational_free_energy(T + dT) - spec.vibrational_free_energy(T - dT)) / (2 * dT)
    assert_allclose(entropy, numerical, rtol=1e-6)


@pytest.mark.parametrize("classical", [False, True])
def test_scalar_temperature_gives_a_scalar_free_energy(classical):
    spec = einstein_spectrum(6.5)
    out = spec.vibrational_free_energy(300.0, classical=classical)
    assert np.isscalar(out) and not isinstance(out, np.ndarray)
    assert spec.vibrational_free_energy(np.array(300.0)).shape == ()
    assert spec.vibrational_free_energy(np.array([300.0, 400.0])).shape == (2,)


def test_arbitrary_unsorted_temperatures_are_answered_in_order():
    spec = einstein_spectrum(6.5)
    T = np.array([900.0, 17.0, 400.0, 0.0])
    assert_allclose(
        spec.vibrational_free_energy(T),
        [spec.vibrational_free_energy(t) for t in T],
        rtol=1e-14,
    )


def test_spectra_hash_and_compare_by_content():
    a = einstein_spectrum(6.5)
    b = einstein_spectrum(6.5)
    assert a is not b
    assert a == b and hash(a) == hash(b)
    assert {a, b} == {a}
    assert a != einstein_spectrum(6.6)
    assert a != einstein_spectrum(6.5, volume=21.0)
    assert a != einstein_spectrum(6.5, energy=-2.0)


# --- PhonopyQuasiHarmonicPhase: the volume minimisation ----------------------------------


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


# --- PhonopyQuasiHarmonicPhase: agreement with the fitted line phase ----------------------


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


# --- PhonopyQuasiHarmonicPhase: dynamical instability ------------------------------------


def test_unstable_volumes_are_dropped_with_a_warning():
    volumes = np.linspace(17.0, 23.0, 7)
    spectra = [einstein_spectrum(5.0, volume=v, energy=vinet_energy(v)) for v in volumes]
    frequencies = np.array(spectra[0].frequencies)
    frequencies[0, 0] = -1.3  # an imaginary branch at the most compressed volume
    spectra[0] = PhononSpectrum(
        frequencies=frequencies, weights=spectra[0].weights, volume=volumes[0],
        energy=vinet_energy(volumes[0]), atoms_per_cell=1,
    )
    with pytest.warns(DynamicalInstabilityWarning, match="dropped 1 of 7"):
        phase = PhonopyQuasiHarmonicPhase("unstable", 0.0, spectra=tuple(spectra))
    assert len(phase.stable_spectra) == 6
    assert len(phase.unstable_spectra) == 1
    assert phase.unstable_spectra[0].volume == volumes[0]
    assert volumes[0] not in phase.sampled_volumes


def test_keeping_an_unstable_volume_changes_the_answer():
    # imaginary modes are silently skipped by the mode sum, so the bad volume still
    # returns a smooth plausible number -- and drags the fit with it
    volumes = np.linspace(17.0, 23.0, 7)
    spectra = [einstein_spectrum(5.0, volume=v, energy=vinet_energy(v)) for v in volumes]
    frequencies = np.array(spectra[0].frequencies)
    frequencies[:, 0] = -1.3
    spectra[0] = PhononSpectrum(
        frequencies=frequencies, weights=spectra[0].weights, volume=volumes[0],
        energy=vinet_energy(volumes[0]), atoms_per_cell=1,
    )
    with pytest.warns(DynamicalInstabilityWarning):
        dropped = PhonopyQuasiHarmonicPhase("dropped", 0.0, spectra=tuple(spectra))
    kept = PhonopyQuasiHarmonicPhase("kept", 0.0, spectra=tuple(spectra), min_frequency=-10.0)
    assert len(kept.stable_spectra) == 7
    assert abs(kept.line_free_energy(600.0) - dropped.line_free_energy(600.0)) > 1e-4


def test_too_few_stable_volumes_is_an_error_not_a_fit():
    volumes = np.linspace(17.0, 23.0, 5)
    spectra = []
    for i, v in enumerate(volumes):
        frequencies = np.full((5, 3), 5.0)
        if i < 2:
            frequencies[:, 0] = -1.3
        spectra.append(
            PhononSpectrum(frequencies=frequencies, weights=np.arange(1, 6), volume=v,
                           energy=vinet_energy(v), atoms_per_cell=1)
        )
    with pytest.raises(ValueError, match="needs at least four"):
        PhonopyQuasiHarmonicPhase("mostly unstable", 0.0, spectra=tuple(spectra))


# --- PhonopyQuasiHarmonicPhase: the extrapolation ceiling --------------------------------


def test_free_energy_is_nan_once_the_equilibrium_volume_leaves_the_sampled_range():
    # a narrow volume window that thermal expansion runs out of
    phase = grueneisen_phase(gamma=3.0, volumes=np.linspace(19.8, 20.6, 5))
    ceiling = phase.max_temperature(upper=4000.0, tolerance=0.5)
    assert 0 < ceiling < 4000.0
    assert np.isfinite(phase.line_free_energy(ceiling - 1.0))
    with pytest.warns(EosExtrapolationWarning, match="outside the sampled range"):
        assert np.isnan(phase.line_free_energy(ceiling + 1.0))
    with pytest.warns(EosExtrapolationWarning):
        assert np.isnan(phase.equilibrium_volume(ceiling + 1.0))


def test_a_wide_enough_volume_range_has_no_ceiling_below_the_bracket():
    phase = grueneisen_phase(gamma=2.0, volumes=np.linspace(17.0, 26.0, 9))
    assert phase.max_temperature(upper=1500.0) == 1500.0


def test_nan_keeps_the_phase_out_of_the_stable_set():
    # calc_phase_diagram picks the stable phase with idxmin, which skips NaN, so a phase
    # above its ceiling drops out of the diagram instead of poisoning it
    solid = grueneisen_phase("solid", gamma=3.0, volumes=np.linspace(19.8, 20.6, 5))
    ceiling = solid.max_temperature(upper=4000.0, tolerance=0.5)
    other = LinePhase("other", 0.0, line_energy=-3.0, line_entropy=1e-4)
    T = np.array([ceiling - 50.0, ceiling + 50.0])
    with pytest.warns(EosExtrapolationWarning):
        df = calc_phase_diagram([solid, other], Ts=T, mu=np.array([0.0]), refine=False)
    stable_above = df.query("T > @ceiling").phase.tolist()
    assert stable_above == ["other"]


# --- PhonopyQuasiHarmonicPhase: construction contract and identity -----------------------


def test_spectra_must_agree_on_the_atom_counts():
    volumes = np.linspace(17.0, 23.0, 5)
    spectra = [einstein_spectrum(5.0, volume=v, energy=vinet_energy(v)) for v in volumes]
    spectra[2] = einstein_spectrum(5.0, volume=volumes[2], energy=vinet_energy(volumes[2]), atoms_per_cell=2)
    with pytest.raises(ValueError, match="same structure"):
        PhonopyQuasiHarmonicPhase("mixed", 0.0, spectra=tuple(spectra))


def test_duplicate_volumes_are_rejected():
    spectra = [einstein_spectrum(5.0, volume=v, energy=vinet_energy(v)) for v in (17.0, 19.0, 21.0, 23.0, 19.0)]
    with pytest.raises(ValueError, match="distinct volumes"):
        PhonopyQuasiHarmonicPhase("duplicate", 0.0, spectra=tuple(spectra))


def test_unknown_equation_of_state_is_rejected():
    with pytest.raises(ValueError, match="eos must be one of"):
        planted_phase(eos="debye")


def test_phases_hash_and_compare_by_content():
    a = planted_phase("p")
    b = planted_phase("p")
    assert a == b and hash(a) == hash(b)
    assert {a, b} == {a}
    assert a != planted_phase("other")
    assert a != planted_phase("p", classical=True)
    assert a != planted_phase("p", eos="murnaghan")
    assert a != planted_phase("p", omega=5.5)


def test_equation_of_state_choice_changes_the_answer():
    vinet = grueneisen_phase("v")
    murnaghan = grueneisen_phase("m", eos="murnaghan")
    assert vinet.line_free_energy(900.0) != murnaghan.line_free_energy(900.0)


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


# --- the live Phonopy path ----------------------------------------------------------------
#
# A real phonopy calculation, with forces from a nearest-neighbour Morse pair potential rather
# than a calculator, so these exercise `from_phonopy` end to end without pulling in a second
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


def morse_crystal(a, supercell=2, mesh=(8, 8, 8)):
    """``(Phonopy, energy, volume)`` for a conventional fcc cell of lattice constant ``a``.

    The cell holds four atoms and reduces to one primitive atom, so this is also the case where
    the spectrum and the energy are normalised by different atom counts.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    atoms = PhonopyAtoms(
        symbols=["Cu"] * 4,
        cell=np.eye(3) * a,
        scaled_positions=[[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    _, energy = morse_forces_and_energy(atoms.cell, atoms.positions)
    phonon = Phonopy(atoms, supercell_matrix=np.eye(3) * supercell, primitive_matrix="auto")
    phonon.generate_displacements(distance=0.01)
    phonon.forces = np.array(
        [morse_forces_and_energy(sc.cell, sc.positions)[0] for sc in phonon.supercells_with_displacements]
    )
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    if mesh is not None:
        phonon.run_mesh(list(mesh), is_gamma_center=True)
    return phonon, energy, phonon.unitcell.volume


def test_from_phonopy_matches_a_hand_built_spectrum():
    phonon, energy, volume = morse_crystal(3.61)

    spec = PhononSpectrum.from_phonopy(phonon, energy=energy)
    assert spec.atoms_per_cell == 4
    assert spec.atoms_per_primitive_cell == 1
    assert_allclose(spec.volume, volume)
    assert_allclose(spec.static_energy_per_atom, energy / 4)
    assert spec.min_frequency > -1e-3  # the Morse fcc crystal is dynamically stable

    manual = PhononSpectrum(
        frequencies=phonon.mesh.frequencies,
        weights=phonon.mesh.weights,
        volume=volume,
        energy=energy,
        atoms_per_cell=4,
    )
    assert spec == manual and hash(spec) == hash(manual)


@pytest.mark.parametrize("classical", [False, True])
def test_from_phonopy_reproduces_phonopys_own_thermal_properties(classical):
    # the spectrum path must return phonopy's number, not a second opinion on it
    phonon, energy, _ = morse_crystal(3.61)
    spec = PhononSpectrum.from_phonopy(phonon, energy=energy)

    T = np.array([0.0, 300.0, 900.0])
    phonon.run_thermal_properties(temperatures=T, classical=classical)
    reference = phonon.thermal_properties.thermal_properties[1] / get_physical_units().EvTokJmol
    assert_allclose(spec.vibrational_free_energy(T, classical=classical), reference, rtol=1e-14, atol=1e-18)


def test_phase_from_phonopy_runs_the_mesh_it_needs():
    pairs = [morse_crystal(a, mesh=None)[:2] for a in np.linspace(3.52, 3.90, 6)]
    assert all(phonon.mesh is None for phonon, _ in pairs)

    phase = PhonopyQuasiHarmonicPhase.from_phonopy("Cu", 0.0, pairs, mesh=[8, 8, 8])
    assert all(phonon.mesh is not None for phonon, _ in pairs)
    assert len(phase.stable_spectra) == 6
    assert phase.stable_spectra[0].atoms_per_primitive_cell == 1
    assert phase.stable_spectra[0].atoms_per_cell == 4

    # Morse force constants soften as the crystal expands, so it expands when heated.  The
    # temperature comes from the phase's own ceiling rather than a guess, since how far this
    # toy potential expands before it runs out of sampled volumes is not something to hard-code
    ceiling = phase.max_temperature(upper=2000.0)
    assert ceiling > 0
    warm = 0.5 * ceiling
    assert phase.line_free_energy(warm) < phase.line_free_energy(0.0)
    assert phase.equilibrium_volume(warm) > phase.equilibrium_volume(0.0)


def test_phase_from_phonopy_reuses_a_mesh_that_has_already_run():
    pairs = [morse_crystal(a)[:2] for a in np.linspace(3.52, 3.90, 6)]
    meshes = [phonon.mesh for phonon, _ in pairs]
    phase = PhonopyQuasiHarmonicPhase.from_phonopy("Cu", 0.0, pairs)
    assert [phonon.mesh for phonon, _ in pairs] == meshes
    assert np.isfinite(phase.line_free_energy(300.0))
