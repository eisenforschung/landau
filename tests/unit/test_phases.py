import numpy as np
from numpy.testing import assert_allclose

from landau.interpolate import SGTE
from landau.interpolate.basic import G_calphad
from landau.phases import LinePhase, TemperatureDependentLinePhase


def _make_line_phase(c=0.25, energy=-1.0, entropy=0.01):
    return LinePhase(name="test", fixed_concentration=c, line_energy=energy, line_entropy=entropy)


def _make_tdlp(c=0.5, T=None, f=None):
    if T is None:
        T = np.linspace(300, 1200, 20)
    if f is None:
        f = G_calphad(T, 1e-4, -2.0, 5e-4)
    return TemperatureDependentLinePhase(
        name="tdlp", fixed_concentration=c, temperatures=T, free_energies=f, interpolator=SGTE(3)
    )


def test_line_phase_semigrand_potential_at_zero_mu():
    T = np.array([300.0, 600.0, 900.0])
    phase = _make_line_phase()
    assert_allclose(phase.semigrand_potential(T, 0.0), phase.line_free_energy(T))


def test_line_phase_semigrand_potential_scalar():
    phase = _make_line_phase()
    result = phase.semigrand_potential(500.0, 0.0)
    assert np.isscalar(result) or result.ndim == 0
    assert_allclose(result, phase.line_free_energy(500.0))


def test_line_phase_concentration_scalar():
    phase = _make_line_phase(c=0.25)
    c = phase.concentration(500.0, 0.0)
    assert c == 0.25


def test_line_phase_concentration_array():
    phase = _make_line_phase(c=0.25)
    T = np.array([300.0, 600.0, 900.0])
    dmu = np.array([0.0, 0.1, -0.1])
    c = phase.concentration(T, dmu)
    assert isinstance(c, np.ndarray)
    assert c.dtype != object
    assert_allclose(c, 0.25)


def test_line_phase_concentration_broadcast():
    # scalar T, array dmu — result should be array, not object
    phase = _make_line_phase(c=0.25)
    dmu = np.linspace(-1, 1, 5)
    c = phase.concentration(500.0, dmu)
    assert isinstance(c, np.ndarray)
    assert c.dtype != object
    assert_allclose(c, 0.25)


def test_temperature_dependent_line_phase_interpolates_through_samples():
    T = np.linspace(300, 1200, 20)
    f = G_calphad(T, 1e-4, -2.0, 5e-4)
    phase = _make_tdlp(T=T, f=f)
    assert_allclose(phase.line_free_energy(T), f, rtol=1e-2)


def test_temperature_dependent_line_phase_hash_stable():
    T = np.linspace(300, 1200, 20)
    f = G_calphad(T, 1e-4, -2.0, 5e-4)
    phase = _make_tdlp(T=T, f=f)
    assert hash(phase) == hash(phase)


def test_temperature_dependent_line_phase_hash_equal_instances():
    T = np.linspace(300, 1200, 20)
    f = G_calphad(T, 1e-4, -2.0, 5e-4)
    phase1 = _make_tdlp(T=T, f=f)
    phase2 = _make_tdlp(T=T, f=f)
    assert hash(phase1) == hash(phase2)

def _make_phase(line_energy=2.0, line_entropy=0.5, fixed_concentration=0.3):
    return LinePhase("test", fixed_concentration, line_energy, line_entropy)


class TestLinePhaseFreeEnergy:
    def test_scalar(self):
        phase = _make_phase()
        result = phase.line_free_energy(300.0)
        assert np.isclose(result, 2.0 - 300.0 * 0.5)
        assert not isinstance(result, np.ndarray)

    def test_array(self):
        phase = _make_phase()
        T = np.array([100.0, 200.0, 300.0])
        result = phase.line_free_energy(T)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert_allclose(result, 2.0 - T * 0.5)

    def test_zero_entropy(self):
        phase = LinePhase("test", 0.5, 3.0)
        T = np.array([0.0, 100.0, 500.0])
        result = phase.line_free_energy(T)
        assert_allclose(result, 3.0)

    def test_scalar_array_parity(self):
        phase = _make_phase()
        T_vals = [100.0, 300.0, 500.0]
        scalar_results = [phase.line_free_energy(T) for T in T_vals]
        array_result = phase.line_free_energy(np.array(T_vals))
        assert_allclose(array_result, scalar_results)

    def test_0d_array(self):
        phase = _make_phase()
        result_scalar = phase.line_free_energy(300.0)
        result_0d = phase.line_free_energy(np.array(300.0))
        assert np.isclose(result_0d, result_scalar)


class TestAbstractLinePhaseConcentration:
    def test_scalar_returns_scalar(self):
        phase = _make_phase()
        result = phase.concentration(300.0, 0.5)
        assert np.isclose(result, 0.3)
        assert not isinstance(result, np.ndarray)

    def test_array_T(self):
        phase = _make_phase()
        T = np.array([100.0, 200.0, 300.0])
        result = phase.concentration(T, 0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert_allclose(result, 0.3)

    def test_array_dmu(self):
        phase = _make_phase()
        dmu = np.array([0.1, 0.2, 0.3])
        result = phase.concentration(300.0, dmu)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert_allclose(result, 0.3)

    def test_both_array(self):
        phase = _make_phase()
        T = np.array([100.0, 200.0])
        dmu = np.array([0.1, 0.2])
        result = phase.concentration(T, dmu)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert_allclose(result, 0.3)

    def test_scalar_array_parity(self):
        phase = _make_phase()
        T_arr = np.array([300.0, 400.0])
        dmu_arr = np.array([0.5, 0.5])
        array_result = phase.concentration(T_arr, dmu_arr)
        for i, (T, dmu) in enumerate(zip(T_arr, dmu_arr)):
            assert np.isclose(array_result[i], phase.concentration(T, dmu))

    def test_0d_array_inputs(self):
        phase = _make_phase()
        result_scalar = phase.concentration(300.0, 0.5)
        result_0d = phase.concentration(np.array(300.0), np.array(0.5))
        assert np.isclose(result_0d, result_scalar)
