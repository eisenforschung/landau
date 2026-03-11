import pytest
import numpy as np
import importlib.util

HAS_ASE = importlib.util.find_spec("ase") is not None

if HAS_ASE:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo, CrystalThermo
    from ase.build import molecule
    from landau.ase_phases import ASEIdealGasPhase, ASEHarmonicPhase, ASECrystalPhase

@pytest.mark.skipif(not HAS_ASE, reason="ASE is not installed")
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

@pytest.mark.skipif(not HAS_ASE, reason="ASE is not installed")
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

@pytest.mark.skipif(not HAS_ASE, reason="ASE is not installed")
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
