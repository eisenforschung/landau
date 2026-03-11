import pytest
import numpy as np

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo, CrystalThermo
    from ase.build import molecule
    from landau.ase_phases import ASEThermoPhase

import itertools

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
def test_ase_thermo_phase_gibbs():
    atoms = molecule('H2')
    ig = IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=atoms, symmetrynumber=2, spin=0)
    phase = ASEThermoPhase("ig_phase", 0.5, thermochem=ig, use_gibbs=True, pressure=1e5)

    assert phase.line_concentration == 0.5
    assert phase.pressure == 1e5
    assert phase.use_gibbs is True

    T_scalar = 300
    G_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(G_scalar) and not isinstance(G_scalar, np.ndarray)

    T_array = np.array([300, 400])
    G_array = phase.line_free_energy(T_array)
    assert G_array.shape == (2,)


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
def test_ase_thermo_phase_helmholtz():
    ht = HarmonicThermo(vib_energies=[0.1])
    phase = ASEThermoPhase("ht_phase", 0.3, thermochem=ht, use_gibbs=False)

    assert phase.line_concentration == 0.3
    assert phase.use_gibbs is False

    T_scalar = 300
    F_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(F_scalar) and not isinstance(F_scalar, np.ndarray)

    T_array = np.array([300, 400])
    F_array = phase.line_free_energy(T_array)
    assert F_array.shape == (2,)


from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@pytest.mark.parametrize("phase_factory", [
    lambda: ASEThermoPhase("ig_phase", 0.5, thermochem=IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=molecule('H2'), symmetrynumber=2, spin=0), use_gibbs=True),
    lambda: ASEThermoPhase("ht_phase", 0.5, thermochem=HarmonicThermo(vib_energies=[0.1]), use_gibbs=False),
    lambda: ASEThermoPhase("ct_phase", 0.5, thermochem=CrystalThermo(phonon_DOS=np.array([1.]), phonon_energies=np.array([0.1]), formula_units=1), use_gibbs=False),
])
@pytest.mark.parametrize("T, dmu, expected_shape", [
    (300.0, 0.0, ()),
    (np.array([300, 400]), 0.0, (2,)),
    (300.0, np.array([-0.1, 0.0, 0.1]), (3,)),
    (np.array([300, 400]), np.array([0.0, 0.1]), (2,)),
    (np.array([300, 400])[:, None], np.array([-0.1, 0.0, 0.1]), (2, 3)),
    (np.array([[300], [400]]), np.array([[-0.1, 0.0, 0.1]]), (2, 3)),
])
def test_ase_semigrand_potential_vectorization(phase_factory, T, dmu, expected_shape):
    phase = phase_factory()
    res = phase.semigrand_potential(T, dmu)
    if expected_shape == ():
        assert np.isscalar(res) and not isinstance(res, np.ndarray)
    else:
        assert res.shape == expected_shape


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@pytest.mark.parametrize("phase_factory", [
    lambda: ASEThermoPhase("ig_phase", 0.5, thermochem=IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=molecule('H2'), symmetrynumber=2, spin=0), use_gibbs=True),
])
@given(
    T=arrays(float, st.integers(1, 3), elements=st.floats(100, 1000)),
    dmu=arrays(float, st.integers(1, 3), elements=st.floats(-1, 1))
)
def test_ase_semigrand_potential_hypothesis(phase_factory, T, dmu):
    phase = phase_factory()
    try:
        res = phase.semigrand_potential(T, dmu)
        expected_shape = np.broadcast_shapes(T.shape, dmu.shape)
        assert res.shape == expected_shape
    except ValueError:
        # Broadcasting might fail if shapes are completely incompatible
        pass
