import pytest
import numpy as np

from hypothesis import given, strategies as st, example
from hypothesis.extra.numpy import arrays, mutually_broadcastable_shapes

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo, CrystalThermo
    from ase.build import molecule
    from landau.ase_phases import ASEThermoPhase

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
def test_ase_thermo_phase_gibbs():
    atoms = molecule('H2')
    ig = IdealGasThermo(vib_energies=[0.1], geometry='linear', atoms=atoms, symmetrynumber=2, spin=0)
    phase = ASEThermoPhase("ig_phase", 0.5, thermochem=ig, pressure=1e5)

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
    phase = ASEThermoPhase("ht_phase", 0.3, thermochem=ht, pressure=None)

    assert phase.line_concentration == 0.3
    assert phase.pressure is None

    T_scalar = 300
    F_scalar = phase.line_free_energy(T_scalar)
    assert np.isscalar(F_scalar) and not isinstance(F_scalar, np.ndarray)

    T_array = np.array([300, 400])
    F_array = phase.line_free_energy(T_array)
    assert F_array.shape == (2,)


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@given(st.data())
def test_ase_semigrand_potential_vectorization(data):
    shapes = data.draw(mutually_broadcastable_shapes(num_shapes=2, max_dims=3, max_side=3))

    T = data.draw(arrays(float, shapes.input_shapes[0], elements=st.floats(100, 1000)))
    dmu = data.draw(arrays(float, shapes.input_shapes[1], elements=st.floats(-1, 1)))

    expected_shape = shapes.result_shape

    # Use a generic HarmonicPhase for broadcasting logic validation
    ht = HarmonicThermo(vib_energies=[0.1])
    phase = ASEThermoPhase("ht_phase", 0.5, thermochem=ht, pressure=None)

    res = phase.semigrand_potential(T, dmu)
    if expected_shape == ():
        assert np.isscalar(res) and not isinstance(res, np.ndarray)
    else:
        assert res.shape == expected_shape

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@pytest.mark.parametrize("T, dmu, expected_shape", [
    (300.0, 0.0, ()),
    (np.array([300, 400]), 0.0, (2,)),
    (300.0, np.array([-0.1, 0.0, 0.1]), (3,)),
    (np.array([300, 400]), np.array([0.0, 0.1]), (2,)),
    (np.array([300, 400])[:, None], np.array([-0.1, 0.0, 0.1]), (2, 3)),
    (np.array([[300], [400]]), np.array([[-0.1, 0.0, 0.1]]), (2, 3)),
])
def test_ase_semigrand_potential_vectorization_examples(T, dmu, expected_shape):
    ht = HarmonicThermo(vib_energies=[0.1])
    phase = ASEThermoPhase("ht_phase", 0.5, thermochem=ht, pressure=None)

    res = phase.semigrand_potential(T, dmu)
    if expected_shape == ():
        assert np.isscalar(res) and not isinstance(res, np.ndarray)
    else:
        assert res.shape == expected_shape
