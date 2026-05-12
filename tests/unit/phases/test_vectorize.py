import pytest
import numpy as np

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, mutually_broadcastable_shapes

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import HarmonicThermo
    from landau.ase_phases import AsePhase

def get_ase_phase():
    if ase_alarm.message is None:
        return AsePhase("ht_phase", 0.5, thermochem=HarmonicThermo(vib_energies=[0.1]), pressure=None)
    return None

@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@pytest.mark.parametrize("phase", [get_ase_phase()])
@given(data=st.data())
def test_semigrand_potential_vectorization(phase, data):
    shapes = data.draw(mutually_broadcastable_shapes(num_shapes=2, max_dims=3, max_side=3))

    T = data.draw(arrays(float, shapes.input_shapes[0], elements=st.floats(100, 1000)))
    dmu = data.draw(arrays(float, shapes.input_shapes[1], elements=st.floats(-1, 1)))

    expected_shape = shapes.result_shape

    res = phase.semigrand_potential(T, dmu)
    if expected_shape == ():
        assert np.isscalar(res) and not isinstance(res, np.ndarray)
    else:
        assert res.shape == expected_shape

    f = phase.line_free_energy(T)
    c = phase.line_concentration
    expected_res = f - c * dmu
    np.testing.assert_allclose(res, expected_res)


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@pytest.mark.parametrize("phase", [get_ase_phase()])
@pytest.mark.parametrize("T, dmu, expected_shape", [
    (300.0, 0.0, ()),
    (np.array([300, 400]), 0.0, (2,)),
    (300.0, np.array([-0.1, 0.0, 0.1]), (3,)),
    (np.array([300, 400]), np.array([0.0, 0.1]), (2,)),
    (np.array([300, 400])[:, None], np.array([-0.1, 0.0, 0.1]), (2, 3)),
    (np.array([[300], [400]]), np.array([[-0.1, 0.0, 0.1]]), (2, 3)),
])
def test_semigrand_potential_vectorization_examples(phase, T, dmu, expected_shape):
    res = phase.semigrand_potential(T, dmu)
    if expected_shape == ():
        assert np.isscalar(res) and not isinstance(res, np.ndarray)
    else:
        assert res.shape == expected_shape

    f = phase.line_free_energy(T)
    c = phase.line_concentration
    expected_res = f - c * dmu
    np.testing.assert_allclose(res, expected_res)
