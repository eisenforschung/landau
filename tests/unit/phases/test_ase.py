import pytest
import numpy as np

from hypothesis import given, strategies as st, example
from hypothesis.extra.numpy import arrays, mutually_broadcastable_shapes

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo
    from ase.build import molecule
    from landau.phases import AsePhase

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
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


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
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
