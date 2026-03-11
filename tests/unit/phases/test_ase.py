import pytest
import numpy as np

from hypothesis import given, strategies as st, example
from hypothesis.extra.numpy import arrays

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo, CrystalThermo
    from ase.build import molecule
    from landau.ase_phases import ASEThermoPhase

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


@pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")
@given(
    T=arrays(float, st.integers(0, 3), elements=st.floats(100, 1000)),
    dmu=arrays(float, st.integers(0, 3), elements=st.floats(-1, 1))
)
@example(T=np.array(300.0), dmu=np.array(0.0))
@example(T=np.array([300, 400]), dmu=np.array(0.0))
@example(T=np.array(300.0), dmu=np.array([-0.1, 0.0, 0.1]))
@example(T=np.array([300, 400]), dmu=np.array([0.0, 0.1]))
@example(T=np.array([300, 400])[:, None], dmu=np.array([-0.1, 0.0, 0.1]))
@example(T=np.array([[300], [400]]), dmu=np.array([[-0.1, 0.0, 0.1]]))
def test_ase_semigrand_potential_vectorization(T, dmu):
    try:
        expected_shape = np.broadcast_shapes(T.shape, dmu.shape)
    except ValueError:
        # Broadcasting might fail if shapes are completely incompatible
        return

    # Use a generic HarmonicPhase for broadcasting logic validation
    ht = HarmonicThermo(vib_energies=[0.1])
    phase = ASEThermoPhase("ht_phase", 0.5, thermochem=ht, use_gibbs=False)

    # Hypothesis generates arrays with 0-dimensions, which isn't the same as native scalars in our `phase.semigrand_potential`.
    # Let's convert true 0d arrays to python scalars to test the scalar logic pathways explicitly, matching the examples.
    T_val = T.item() if T.ndim == 0 else T
    dmu_val = dmu.item() if dmu.ndim == 0 else dmu

    res = phase.semigrand_potential(T_val, dmu_val)
    if expected_shape == ():
        assert np.isscalar(res) and not isinstance(res, np.ndarray)
    else:
        assert res.shape == expected_shape
