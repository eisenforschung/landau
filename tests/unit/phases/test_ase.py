import pytest
import numpy as np

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo
    from ase.build import molecule
    from landau.phases.asewrapper import AsePhase

pytestmark = pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")


def test_ase_thermo_phase_gibbs():
    atoms = molecule('H2')
    ig = IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=atoms, symmetrynumber=2, spin=0)
    phase = AsePhase("ig_phase", 0.5, thermochem=ig, pressure=1e5)

    assert phase.line_concentration == 0.5
    assert phase.pressure == 1e5

    T_scalar = 300
    G_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(G_scalar) and not isinstance(G_scalar, np.ndarray)

    T_array = np.array([300, 400])
    G_array = phase.line_free_energy(T_array)
    assert G_array.shape == (2,)


def test_ase_thermo_phase_helmholtz():
    ht = HarmonicThermo(vib_energies=[0.1])
    phase = AsePhase("ht_phase", 0.3, thermochem=ht, pressure=None)

    assert phase.line_concentration == 0.3
    assert phase.pressure is None

    T_scalar = 300
    F_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(F_scalar) and not isinstance(F_scalar, np.ndarray)

    T_array = np.array([300, 400])
    F_array = phase.line_free_energy(T_array)
    assert F_array.shape == (2,)


def test_ase_phase_equality_harmonic():
    # ASE's HarmonicThermo defaults to identity equality, so identically
    # constructed instances are not ==.  AsePhase must look past that.
    ht1 = HarmonicThermo(vib_energies=[0.1])
    ht2 = HarmonicThermo(vib_energies=[0.1])
    assert ht1 is not ht2
    assert ht1 != ht2

    p1 = AsePhase("ht", 0.3, thermochem=ht1)
    p2 = AsePhase("ht", 0.3, thermochem=ht2)
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_ase_phase_equality_idealgas():
    ig1 = IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=molecule('H2'), symmetrynumber=2, spin=0)
    ig2 = IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=molecule('H2'), symmetrynumber=2, spin=0)

    p1 = AsePhase("ig", 0.5, thermochem=ig1, pressure=1e5)
    p2 = AsePhase("ig", 0.5, thermochem=ig2, pressure=1e5)
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_ase_phase_inequality():
    ht = HarmonicThermo(vib_energies=[0.1])
    base = AsePhase("ht", 0.3, thermochem=ht)
    assert base != AsePhase("other", 0.3, thermochem=ht)
    assert base != AsePhase("ht", 0.4, thermochem=ht)
    assert base != AsePhase("ht", 0.3, thermochem=ht, pressure=1e5)
    assert base != AsePhase("ht", 0.3, thermochem=HarmonicThermo(vib_energies=[0.2]))


def test_ase_phase_set_dedup():
    p1 = AsePhase("ht", 0.3, thermochem=HarmonicThermo(vib_energies=[0.1]))
    p2 = AsePhase("ht", 0.3, thermochem=HarmonicThermo(vib_energies=[0.1]))
    assert {p1, p2} == {p1}
