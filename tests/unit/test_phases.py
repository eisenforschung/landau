import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import Boltzmann, eV

from landau.interpolate import SGTE
from landau.interpolate.basic import G_calphad
from landau.phases import IdealSolution, LinePhase, RegularSolution, TemperatureDependentLinePhase
from landau.phases import _scalarize

kB = Boltzmann / eV


# --- _scalarize tests ---


def test_scalarize_zero_d_float_array_returns_python_float():
    out = _scalarize(np.array(3.5))
    assert type(out) is float
    assert out == 3.5


def test_scalarize_zero_d_int_array_returns_python_int():
    out = _scalarize(np.array(7))
    assert type(out) is int
    assert out == 7


def test_scalarize_one_d_array_passes_through_unchanged():
    arr = np.array([1.0, 2.0, 3.0])
    out = _scalarize(arr)
    assert out is arr


def test_scalarize_python_float_passes_through_unchanged():
    out = _scalarize(2.5)
    assert type(out) is float
    assert out == 2.5


def test_scalarize_non_array_with_no_ndim_passes_through():
    payload = [1, 2, 3]
    out = _scalarize(payload)
    assert out is payload


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


def test_line_phase_free_energy_scalar():
    phase = _make_line_phase(energy=2.0, entropy=0.5, c=0.3)
    result = phase.line_free_energy(300.0)
    assert np.isclose(result, 2.0 - 300.0 * 0.5)
    assert not isinstance(result, np.ndarray)


def test_line_phase_free_energy_array():
    phase = _make_line_phase(energy=2.0, entropy=0.5, c=0.3)
    T = np.array([100.0, 200.0, 300.0])
    result = phase.line_free_energy(T)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_allclose(result, 2.0 - T * 0.5)


def test_line_phase_free_energy_zero_entropy():
    phase = LinePhase("test", 0.5, 3.0)
    T = np.array([0.0, 100.0, 500.0])
    assert_allclose(phase.line_free_energy(T), 3.0)


def test_line_phase_free_energy_scalar_array_parity():
    phase = _make_line_phase(energy=2.0, entropy=0.5, c=0.3)
    T_vals = [100.0, 300.0, 500.0]
    scalar_results = [phase.line_free_energy(T) for T in T_vals]
    assert_allclose(phase.line_free_energy(np.array(T_vals)), scalar_results)


def test_line_phase_free_energy_0d_array():
    phase = _make_line_phase(energy=2.0, entropy=0.5, c=0.3)
    assert np.isclose(phase.line_free_energy(np.array(300.0)), phase.line_free_energy(300.0))


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
    phase = _make_line_phase(c=0.3, energy=2.0, entropy=0.5)
    result = phase.concentration(300.0, 0.5)
    assert np.isclose(result, 0.3)
    assert not isinstance(result, np.ndarray)


def test_line_phase_concentration_array_T():
    phase = _make_line_phase(c=0.3, energy=2.0, entropy=0.5)
    T = np.array([100.0, 200.0, 300.0])
    result = phase.concentration(T, 0.5)
    assert isinstance(result, np.ndarray)
    assert result.dtype != object
    assert result.shape == (3,)
    assert_allclose(result, 0.3)


def test_line_phase_concentration_array_dmu():
    # scalar T, array dmu — result should be array, not object
    phase = _make_line_phase(c=0.25)
    dmu = np.linspace(-1, 1, 5)
    result = phase.concentration(500.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.dtype != object
    assert result.shape == (5,)
    assert_allclose(result, 0.25)


def test_line_phase_concentration_both_array():
    phase = _make_line_phase(c=0.3, energy=2.0, entropy=0.5)
    T = np.array([100.0, 200.0, 300.0])
    dmu = np.array([0.0, 0.1, -0.1])
    result = phase.concentration(T, dmu)
    assert isinstance(result, np.ndarray)
    assert result.dtype != object
    assert result.shape == (3,)
    assert_allclose(result, 0.3)


def test_line_phase_concentration_scalar_array_parity():
    phase = _make_line_phase(c=0.3, energy=2.0, entropy=0.5)
    T_arr = np.array([300.0, 400.0])
    dmu_arr = np.array([0.5, 0.5])
    array_result = phase.concentration(T_arr, dmu_arr)
    for i, (T, dmu) in enumerate(zip(T_arr, dmu_arr)):
        assert np.isclose(array_result[i], phase.concentration(T, dmu))


def test_line_phase_concentration_0d_array():
    phase = _make_line_phase(c=0.3, energy=2.0, entropy=0.5)
    assert np.isclose(
        phase.concentration(np.array(300.0), np.array(0.5)),
        phase.concentration(300.0, 0.5),
    )


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


# SGTE(3) has the same functional form as G_calphad with 3 params — fit is exact
_TDLP_ATOL = 1e-6


def test_tdlp_interpolation_heldout_accuracy():
    """Interpolation at held-out temperatures matches the analytic G_calphad."""
    T_train = np.linspace(300, 1200, 20)
    f_train = G_calphad(T_train, 1e-4, -2.0, 5e-4)
    phase = _make_tdlp(T=T_train, f=f_train)
    T_test = np.linspace(310, 1190, 50)
    expected = G_calphad(T_test, 1e-4, -2.0, 5e-4)
    assert_allclose(phase.line_free_energy(T_test), expected, atol=_TDLP_ATOL)


def test_tdlp_semigrand_potential_scalar():
    phase = _make_tdlp(c=0.5)
    result = phase.semigrand_potential(500.0, 0.1)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)
    expected = phase.line_free_energy(500.0) - 0.5 * 0.1
    assert_allclose(result, expected, atol=_TDLP_ATOL)


def test_tdlp_semigrand_potential_array_T():
    phase = _make_tdlp(c=0.5)
    T = np.array([400.0, 700.0, 1000.0])
    result = phase.semigrand_potential(T, 0.2)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = phase.line_free_energy(T) - 0.5 * 0.2
    assert_allclose(result, expected, atol=_TDLP_ATOL)


def test_tdlp_semigrand_potential_array_dmu():
    phase = _make_tdlp(c=0.5)
    dmu = np.array([-0.2, 0.0, 0.2])
    result = phase.semigrand_potential(500.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = phase.line_free_energy(500.0) - 0.5 * dmu
    assert_allclose(result, expected, atol=_TDLP_ATOL)


def test_tdlp_concentration_scalar():
    phase = _make_tdlp(c=0.4)
    result = phase.concentration(500.0, 0.1)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)
    assert_allclose(result, 0.4, atol=_TDLP_ATOL)


def test_tdlp_concentration_array_T():
    phase = _make_tdlp(c=0.4)
    T = np.array([400.0, 700.0, 1000.0])
    result = phase.concentration(T, 0.1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_allclose(result, 0.4, atol=_TDLP_ATOL)


def test_tdlp_concentration_array_dmu():
    phase = _make_tdlp(c=0.4)
    dmu = np.array([-0.2, 0.0, 0.2])
    result = phase.concentration(500.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_allclose(result, 0.4, atol=_TDLP_ATOL)


# --- IdealSolution tests ---

_IDEAL_ATOL = 1e-8


def _make_ideal_solution(e1=0.0, s1=0.0, e2=0.0, s2=0.0):
    p0 = LinePhase("A", 0, e1, s1)
    p1 = LinePhase("B", 1, e2, s2)
    return IdealSolution("sol", p0, p1)


def test_ideal_concentration_scalar():
    """With f1=f2=0, dmu=0: c = 0.5 by symmetry."""
    sol = _make_ideal_solution()
    result = sol.concentration(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)
    assert_allclose(result, 0.5, atol=_IDEAL_ATOL)


def test_ideal_concentration_analytic():
    """c = 1 / (1 + exp(-dmu/kB/T)) when f1=f2=0."""
    sol = _make_ideal_solution()
    T = np.array([500.0, 1000.0, 1500.0])
    dmu = np.array([-0.1, 0.0, 0.1])
    result = sol.concentration(T, dmu)
    expected = 1 / (1 + np.exp(-dmu / kB / T))
    assert_allclose(result, expected, atol=_IDEAL_ATOL)


def test_ideal_concentration_midpoint():
    """At dmu = f2 - f1, concentration = 0.5 regardless of T."""
    sol = _make_ideal_solution(e1=0.0, e2=0.3)
    result = sol.concentration(1000.0, 0.3)
    assert_allclose(result, 0.5, atol=1e-6)


def test_ideal_concentration_array_T():
    sol = _make_ideal_solution()
    T = np.array([500.0, 1000.0, 1500.0])
    result = sol.concentration(T, 0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_allclose(result, 0.5, atol=_IDEAL_ATOL)


def test_ideal_concentration_array_dmu():
    sol = _make_ideal_solution()
    dmu = np.array([-0.5, 0.0, 0.5])
    result = sol.concentration(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = 1 / (1 + np.exp(-dmu / kB / 1000.0))
    assert_allclose(result, expected, atol=_IDEAL_ATOL)


def test_ideal_concentration_both_array():
    sol = _make_ideal_solution()
    T = np.array([500.0, 1000.0, 1500.0])
    dmu = np.array([-0.1, 0.0, 0.1])
    result = sol.concentration(T, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = 1 / (1 + np.exp(-dmu / kB / T))
    assert_allclose(result, expected, atol=_IDEAL_ATOL)


def test_ideal_semigrand_potential_scalar():
    """With f1=f2=0, dmu=0: phi = -kB*T*log(2)."""
    sol = _make_ideal_solution()
    T = 1000.0
    result = sol.semigrand_potential(T, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)
    assert_allclose(result, -kB * T * np.log(2), atol=_IDEAL_ATOL)


def test_ideal_semigrand_potential_analytic():
    """phi = -kB*T*log(1 + exp(dmu/kB/T)) when f1=f2=0."""
    sol = _make_ideal_solution()
    T = np.array([500.0, 1000.0, 1500.0])
    dmu = np.array([-0.1, 0.0, 0.1])
    result = sol.semigrand_potential(T, dmu)
    expected = -kB * T * np.log(1 + np.exp(dmu / kB / T))
    assert_allclose(result, expected, atol=_IDEAL_ATOL)


def test_ideal_semigrand_potential_array_T():
    sol = _make_ideal_solution()
    T = np.array([500.0, 1000.0, 1500.0])
    result = sol.semigrand_potential(T, 0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_allclose(result, -kB * T * np.log(2), atol=_IDEAL_ATOL)


def test_ideal_semigrand_potential_array_dmu():
    sol = _make_ideal_solution()
    dmu = np.array([-0.1, 0.0, 0.1])
    result = sol.semigrand_potential(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = -kB * 1000.0 * np.log(1 + np.exp(dmu / kB / 1000.0))
    assert_allclose(result, expected, atol=_IDEAL_ATOL)


def test_ideal_concentration_from_semigrand_potential():
    """Verify Gibbs-Duhem: c = -d(phi)/d(dmu)."""
    sol = _make_ideal_solution()
    T = 1000.0
    dmu = np.linspace(-0.5, 0.5, 200)
    phi = sol.semigrand_potential(T, dmu)
    c_numeric = -np.gradient(phi, dmu)
    c_direct = sol.concentration(T, dmu)
    assert_allclose(c_numeric, c_direct, atol=1e-3)


# --- RegularSolution tests ---

_RS_ATOL = 1e-8


def _make_regular_solution():
    """Symmetric three-phase RegularSolution with attractive interaction (num_coeffs=1)."""
    p0 = LinePhase("A", 0, 0.0)
    pm = LinePhase("mid", 0.5, -0.5)
    p1 = LinePhase("B", 1, 0.0)
    return RegularSolution("sol", [p0, pm, p1])


def test_regular_solution_semigrand_potential_scalar():
    sol = _make_regular_solution()
    result = sol.semigrand_potential(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)


def test_regular_solution_semigrand_potential_array_dmu():
    sol = _make_regular_solution()
    dmu = np.linspace(-1.0, 1.0, 15)
    result = sol.semigrand_potential(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (15,)


def test_regular_solution_concentration_scalar():
    sol = _make_regular_solution()
    result = sol.concentration(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)
    assert 0.0 <= float(result) <= 1.0


def test_regular_solution_concentration_array_dmu():
    sol = _make_regular_solution()
    dmu = np.linspace(-1.0, 1.0, 15)
    result = sol.concentration(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (15,)
    assert np.all((result >= 0) & (result <= 1))


def test_regular_solution_concentration_symmetric_at_zero():
    """At dmu=0 with equal terminal energies, concentration = 0.5 by symmetry."""
    sol = _make_regular_solution()  # f0 = f1 = 0
    result = sol.concentration(1000.0, 0.0)
    assert_allclose(result, 0.5, atol=_RS_ATOL)


def test_regular_solution_concentration_midpoint_asymmetric():
    """At dmu = f1 - f0, the free-energy landscape is symmetric and concentration = 0.5."""
    p0 = LinePhase("A", 0, 0.3)
    pm = LinePhase("mid", 0.5, -0.5)
    p1 = LinePhase("B", 1, 0.0)
    sol = RegularSolution("sol", [p0, pm, p1])
    result = sol.concentration(1000.0, 0.0 - 0.3)  # dmu = f1 - f0 = -0.3
    assert_allclose(result, 0.5, atol=_RS_ATOL)


def test_regular_solution_concentration_monotone():
    """Concentration increases strictly with dmu; fails for constant implementations."""
    sol = _make_regular_solution()
    dmu = np.linspace(-1.0, 1.0, 20)
    c = sol.concentration(1000.0, dmu)
    assert np.all(np.diff(c) > 0)
