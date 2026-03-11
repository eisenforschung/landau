import pytest
import numpy as np

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo, CrystalThermo
    from ase.build import molecule
    from landau.ase_phases import ASEIdealGasPhase, ASEHarmonicPhase, ASECrystalPhase

import itertools

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
def test_ase_ideal_gas_phase():
    atoms = molecule('H2')
    ig = IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=atoms, symmetrynumber=2, spin=0)
    phase = ASEIdealGasPhase("ig_phase", 0.5, thermochem=ig, pressure=1e5)

    assert phase.line_concentration == 0.5
    assert phase.pressure == 1e5

    T_scalar = 300
    G_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(G_scalar) or G_scalar.ndim == 0

    T_array = np.array([300, 400])
    G_array = phase.line_free_energy(T_array)
    assert G_array.shape == (2,)

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
def test_ase_harmonic_phase():
    ht = HarmonicThermo(vib_energies=[0.1])
    phase = ASEHarmonicPhase("ht_phase", 0.3, thermochem=ht)

    assert phase.line_concentration == 0.3

    T_scalar = 300
    F_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(F_scalar) or F_scalar.ndim == 0

    T_array = np.array([300, 400])
    F_array = phase.line_free_energy(T_array)
    assert F_array.shape == (2,)


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@pytest.mark.parametrize("phase_factory", [
    lambda: ASEIdealGasPhase("ig_phase", 0.5, thermochem=IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=molecule('H2'), symmetrynumber=2, spin=0)),
    lambda: ASEHarmonicPhase("ht_phase", 0.5, thermochem=HarmonicThermo(vib_energies=[0.1])),
    lambda: ASECrystalPhase("ct_phase", 0.5, thermochem=CrystalThermo(phonon_DOS=np.array([1.]), phonon_energies=np.array([0.1]), formula_units=1)),
])
def test_ase_semigrand_potential_vectorization(phase_factory):
    phase = phase_factory()

    T_scalar = 300.0
    dmu_scalar = 0.0

    T_1d = np.array([300, 400])
    dmu_1d = np.array([-0.1, 0.0, 0.1])

    T_2d = np.array([[300], [400]])
    dmu_2d = np.array([[-0.1, 0.0, 0.1]])

    # scalar, scalar
    res = phase.semigrand_potential(T_scalar, dmu_scalar)
    assert np.isscalar(res) or res.ndim == 0

    # 1d, scalar
    res = phase.semigrand_potential(T_1d, dmu_scalar)
    assert res.shape == (2,)

    # scalar, 1d
    res = phase.semigrand_potential(T_scalar, dmu_1d)
    assert res.shape == (3,)

    # 1d, 1d (same length)
    res = phase.semigrand_potential(T_1d, np.array([0.0, 0.1]))
    assert res.shape == (2,)

    # broadcasting 1d, 1d -> 2d
    res = phase.semigrand_potential(T_1d[:, None], dmu_1d)
    assert res.shape == (2, 3)

    # broadcasting 2d, 2d -> 2d
    res = phase.semigrand_potential(T_2d, dmu_2d)
    assert res.shape == (2, 3)


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
def test_ase_crystal_phase():
    ct = CrystalThermo(phonon_DOS=np.array([1.]), phonon_energies=np.array([0.1]), formula_units=1)
    phase = ASECrystalPhase("ct_phase", 0.8, thermochem=ct)

    assert phase.line_concentration == 0.8

    T_scalar = 300
    F_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(F_scalar) or F_scalar.ndim == 0

    T_array = np.array([300, 400])
    F_array = phase.line_free_energy(T_array)
    assert F_array.shape == (2,)
