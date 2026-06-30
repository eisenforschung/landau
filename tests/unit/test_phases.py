import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.constants import Boltzmann, eV

from landau.interpolate import PolyFit, SGTE
from landau.interpolate.basic import G_calphad
from landau.phases import IdealSolution, InterpolatingPhase, LinePhase, RegularSolution, SlowInterpolatingPhase, FastInterpolatingPhase, TemperatureDependentLinePhase
from landau.phases.pointdefects import (
    AbstractPointDefectSublattice,
    ConstantPointDefect,
    LowTemperatureExpansionSublattice,
    PointDefectSublattice,
    PointDefectedPhase,
)
from landau.phases import S, _scalarize

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


# --- S(c) ideal-mixing entropy tests ---


def test_S_endpoints_are_zero():
    assert S(0.0) == 0.0
    assert S(1.0) == 0.0


def test_S_midpoint_is_kB_ln_two():
    assert_allclose(S(0.5), kB * np.log(2), atol=1e-15)


def test_S_symmetric_under_c_to_one_minus_c():
    c = np.linspace(0.0, 1.0, 11)
    assert_allclose(S(c), S(1 - c), atol=1e-15)


def test_S_scalar_in_scalar_out():
    out = S(0.3)
    assert np.isscalar(out) or (isinstance(out, np.ndarray) and out.ndim == 0)


def test_S_array_in_array_out():
    c = np.linspace(0.0, 1.0, 5)
    out = S(c)
    assert isinstance(out, np.ndarray)
    assert out.shape == c.shape


def test_S_non_negative_on_unit_interval():
    c = np.linspace(0.0, 1.0, 101)
    assert np.all(S(c) >= 0.0)


def test_S_agrees_with_closed_form_on_interior():
    c = np.linspace(0.05, 0.95, 19)
    closed_form = -kB * (c * np.log(c) + (1 - c) * np.log(1 - c))
    assert_allclose(S(c), closed_form, atol=1e-15)


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


# --- InterpolatingPhase / SlowInterpolatingPhase tests ---

# InterpolatingPhase uses a 100-point grid + one Newton-Raphson refinement step.
# SlowInterpolatingPhase uses scipy.optimize.brute with fmin finish — effectively exact.
# The cross-validation tolerance reflects the grid error of the fast variant.
_INTERP_ATOL = 5e-3


def _make_interp_line_phases():
    """Three line phases spanning [0, 1] with a concave-down midpoint."""
    return [
        LinePhase("A", fixed_concentration=0.0, line_energy=0.0),
        LinePhase("mid", fixed_concentration=0.5, line_energy=-0.3),
        LinePhase("B", fixed_concentration=1.0, line_energy=0.0),
    ]


def _make_interpolating_phase():
    return InterpolatingPhase("sol", _make_interp_line_phases())


def _make_slow_interpolating_phase():
    return SlowInterpolatingPhase("sol", _make_interp_line_phases())


def test_interpolating_phase_semigrand_potential_scalar():
    result = _make_interpolating_phase().semigrand_potential(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)


def test_interpolating_phase_semigrand_potential_array_dmu():
    dmu = np.linspace(-0.5, 0.5, 11)
    result = _make_interpolating_phase().semigrand_potential(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (11,)


def test_interpolating_phase_semigrand_potential_array_T():
    T = np.array([500.0, 1000.0, 1500.0])
    result = _make_interpolating_phase().semigrand_potential(T, 0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_interpolating_phase_concentration_scalar():
    result = _make_interpolating_phase().concentration(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)
    assert 0.0 <= float(result) <= 1.0


def test_interpolating_phase_concentration_array_dmu():
    dmu = np.linspace(-0.5, 0.5, 11)
    result = _make_interpolating_phase().concentration(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (11,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_interpolating_phase_concentration_monotone():
    """Concentration is non-decreasing in dmu for a non-degenerate landscape."""
    dmu = np.linspace(-0.5, 0.5, 25)
    c = _make_interpolating_phase().concentration(1000.0, dmu)
    assert np.all(np.diff(c) >= 0)


def test_interpolating_phase_gibbs_duhem():
    """c = -d(phi)/d(dmu) from the Legendre transform."""
    phase = _make_interpolating_phase()
    dmu = np.linspace(-0.4, 0.4, 400)
    phi = phase.semigrand_potential(1000.0, dmu)
    c_numeric = -np.gradient(phi, dmu)
    c_direct = phase.concentration(1000.0, dmu)
    assert_allclose(c_numeric[5:-5], c_direct[5:-5], atol=_INTERP_ATOL)


def test_slow_interpolating_phase_semigrand_potential_scalar():
    result = _make_slow_interpolating_phase().semigrand_potential(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)


def test_slow_interpolating_phase_concentration_scalar():
    result = _make_slow_interpolating_phase().concentration(1000.0, 0.0)
    assert np.isscalar(result) or not isinstance(result, np.ndarray)
    assert 0.0 <= float(result) <= 1.0


def test_slow_interpolating_phase_concentration_array_dmu():
    dmu = np.linspace(-0.5, 0.5, 7)
    result = _make_slow_interpolating_phase().concentration(1000.0, dmu)
    assert isinstance(result, np.ndarray)
    assert result.shape == (7,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_slow_interpolating_phase_gibbs_duhem():
    """c = -d(phi)/d(dmu) from the Legendre transform."""
    phase = _make_slow_interpolating_phase()
    dmu = np.linspace(-0.4, 0.4, 200)
    phi = phase.semigrand_potential(1000.0, dmu)
    c_numeric = -np.gradient(phi, dmu)
    c_direct = phase.concentration(1000.0, dmu)
    assert_allclose(c_numeric[5:-5], c_direct[5:-5], atol=_INTERP_ATOL)


def test_interpolating_vs_slow_semigrand_potential():
    """InterpolatingPhase and SlowInterpolatingPhase agree on semigrand_potential."""
    fast = _make_interpolating_phase()
    slow = _make_slow_interpolating_phase()
    dmu = np.linspace(-0.4, 0.4, 11)
    assert_allclose(fast.semigrand_potential(1000.0, dmu), slow.semigrand_potential(1000.0, dmu), atol=_INTERP_ATOL)


def test_interpolating_vs_slow_concentration():
    """InterpolatingPhase and SlowInterpolatingPhase agree on concentration."""
    fast = _make_interpolating_phase()
    slow = _make_slow_interpolating_phase()
    dmu = np.linspace(-0.4, 0.4, 11)
    assert_allclose(fast.concentration(1000.0, dmu), slow.concentration(1000.0, dmu), atol=_INTERP_ATOL)


def test_slow_interpolating_phase_explicit_concentration_range_honoured():
    """Explicitly passed concentration_range is not overridden by maximum_extrapolation=0."""
    phases = [
        LinePhase("A", fixed_concentration=0.2, line_energy=0.0),
        LinePhase("mid", fixed_concentration=0.5, line_energy=-0.3),
        LinePhase("B", fixed_concentration=0.8, line_energy=0.0),
    ]
    explicit = (0.1, 0.9)
    phase = SlowInterpolatingPhase("sol", phases, concentration_range=explicit)
    assert phase.concentration_range == explicit


def test_slow_interpolating_phase_concentration_range_and_extrapolation_exclusive():
    """Passing both concentration_range and maximum_extrapolation raises ValueError."""
    phases = [
        LinePhase("A", fixed_concentration=0.2, line_energy=0.0),
        LinePhase("B", fixed_concentration=0.8, line_energy=0.0),
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        SlowInterpolatingPhase("sol", phases, concentration_range=(0.1, 0.9), maximum_extrapolation=0.05)


# --- FastInterpolatingPhase tests ---

# FastInterpolatingPhase solves the same minimization as SlowInterpolatingPhase
# (a vectorized logit-Newton polish replaces the per-scalar brute), so it is
# validated against the true minimum on a dense grid, not against brute's coarse
# 20-point output. The tolerance is tighter than the InterpolatingPhase grid
# tolerance because the Newton polish converges to the true optimum.
_FAST_ATOL = 1e-5


def _fast_phases():
    """A PolyFit interior phase and a RedlichKister full-range phase."""
    poly = FastInterpolatingPhase("poly", [
        LinePhase("a", 0.30, -2.30),
        LinePhase("b", 0.40, -2.50),
        LinePhase("c", 0.50, -2.60),
        LinePhase("d", 0.60, -2.55),
        LinePhase("e", 0.70, -2.40),
    ])
    rk = FastInterpolatingPhase("rk", [
        LinePhase("A", 0.0, 0.0),
        LinePhase("q1", 0.25, -0.20),
        LinePhase("mid", 0.50, -0.30),
        LinePhase("q3", 0.75, -0.18),
        LinePhase("B", 1.0, 0.0),
    ])
    return poly, rk


def _gold_min(phase, T, dmu):
    """True (phi, c) at the global minimum of f(c) - dmu*c on a dense grid."""
    fe = phase._get_interpolation(float(T))
    a, b = phase.concentration_range
    cs = np.linspace(a, b, 80001)
    val = fe(cs)[:, None] - T * S(cs)[:, None] - np.atleast_1d(dmu)[None, :] * cs[:, None]
    i = val.argmin(axis=0)
    return val[i, np.arange(val.shape[1])], cs[i]


def test_fast_interpolating_phase_scalar_contract():
    poly, _ = _fast_phases()
    phi = poly.semigrand_potential(1000.0, 0.1)
    c = poly.concentration(1000.0, 0.1)
    assert not isinstance(phi, np.ndarray)
    assert not isinstance(c, np.ndarray)
    assert 0.0 <= float(c) <= 1.0


def test_fast_interpolating_phase_array_dmu_shape():
    poly, _ = _fast_phases()
    dmu = np.linspace(-3.0, 3.0, 17)
    phi = poly.semigrand_potential(1000.0, dmu)
    c = poly.concentration(1000.0, dmu)
    assert phi.shape == (17,)
    assert c.shape == (17,)
    assert np.all((c >= 0.0) & (c <= 1.0))


def test_fast_interpolating_phase_array_T_shape():
    _, rk = _fast_phases()
    T = np.array([500.0, 1000.0, 1500.0])
    phi = rk.semigrand_potential(T, 0.2)
    assert phi.shape == (3,)


def test_fast_interpolating_phase_concentration_monotone():
    """Concentration is non-decreasing in dmu."""
    _, rk = _fast_phases()
    dmu = np.linspace(-4.0, 4.0, 51)
    c = rk.concentration(1000.0, dmu)
    assert np.all(np.diff(c) >= -1e-9)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("T", [400.0, 1000.0, 1700.0])
def test_fast_interpolating_phase_matches_true_minimum(which, T):
    """phi and c reproduce the dense-grid global minimum, including saturation."""
    phase = _fast_phases()[which]
    dmu = np.linspace(-10.0, 10.0, 101)
    gphi, gc = _gold_min(phase, T, dmu)
    phi = np.asarray(phase.semigrand_potential(T, dmu))
    c = np.asarray(phase.concentration(T, dmu))
    assert not np.isnan(phi).any()
    assert_allclose(phi, gphi, atol=_FAST_ATOL)
    assert_allclose(c, gc, atol=1e-3)


def test_fast_interpolating_phase_gibbs_duhem():
    """c = -d(phi)/d(dmu) from the Legendre transform.

    Tolerance is set by the finite-difference gradient over a smooth interior
    window, not by Fast's own accuracy.
    """
    _, rk = _fast_phases()
    dmu = np.linspace(-0.5, 0.5, 400)
    phi = rk.semigrand_potential(1000.0, dmu)
    c_numeric = -np.gradient(phi, dmu)
    c_direct = rk.concentration(1000.0, dmu)
    assert_allclose(c_numeric[5:-5], c_direct[5:-5], atol=_INTERP_ATOL)


def test_fast_concentration_consistent_with_phi_gradient():
    """On a convex (jump-free) phase, c equals -d(phi)/d(dmu) everywhere.

    A convex free energy has no miscibility gap, so phi(dmu) is smooth and the
    Legendre relation must hold across the whole window, not just within a
    branch. The tolerance is the central-difference gradient error of the test.
    """
    convex = FastInterpolatingPhase("cvx", [
        LinePhase("A", 0.0, 0.0),
        LinePhase("mid", 0.5, -0.6),
        LinePhase("B", 1.0, 0.0),
    ])
    dmu = np.linspace(-2.0, 2.0, 600)
    phi = convex.semigrand_potential(1000.0, dmu)
    c = convex.concentration(1000.0, dmu)
    assert_allclose(-np.gradient(phi, dmu)[5:-5], c[5:-5], atol=1e-5)


@pytest.mark.parametrize("T", [400.0, 1200.0])
def test_fast_concentration_is_exact_stationary_point(T):
    """f'(c) = dmu at every interior solution (so c = -d(phi)/d(dmu) exactly).

    The analytic first derivative makes the located concentration the exact
    stationary point; only points pinned to a range boundary are excluded.
    """
    poly = _fast_phases()[0]
    a, b = poly.concentration_range
    fe = poly._get_interpolation(T)
    dmu = np.linspace(-3.0, 3.0, 200)
    c = np.asarray(poly.concentration(T, dmu))
    # f'(c) = fe'(c) - T*S'(c) = fe'(c) + kB*T*logit(c)
    kB = Boltzmann / eV
    fprime = fe.deriv()(c) + kB * T * (np.log(c) - np.log1p(-c))
    interior = (c > a + 1e-9) & (c < b - 1e-9)
    assert interior.any()
    assert np.abs(fprime[interior] - dmu[interior]).max() < 1e-8


def test_fast_matches_slow_where_brute_is_reliable():
    """On the smooth symmetric fixture, Fast and Slow agree to brute's tolerance."""
    pts = _make_interp_line_phases()
    fast = FastInterpolatingPhase("sol", pts)
    slow = SlowInterpolatingPhase("sol", pts)
    dmu = np.linspace(-0.4, 0.4, 21)
    assert_allclose(fast.semigrand_potential(1000.0, dmu),
                    slow.semigrand_potential(1000.0, dmu), atol=1e-4)
    assert_allclose(fast.concentration(1000.0, dmu),
                    slow.concentration(1000.0, dmu), atol=1e-3)


def test_fast_finds_deep_well_global_minimum():
    """A deep off-centre well whose global minimum a coarse grid can miss.

    Fast must still land in the true global basin (verified on a dense grid).
    """
    pts = [
        LinePhase("A", 0.0, -2.221),
        LinePhase("b", 0.551, -1.048),
        LinePhase("c", 0.76, -2.957),
        LinePhase("d", 0.845, -0.505),
        LinePhase("B", 1.0, -0.491),
    ]
    phase = FastInterpolatingPhase("deep", pts)
    dmu = np.linspace(-12.0, 12.0, 121)
    for T in (300.0, 1500.0):
        gphi, gc = _gold_min(phase, T, dmu)
        assert_allclose(np.asarray(phase.semigrand_potential(T, dmu)), gphi, atol=_FAST_ATOL)
        assert_allclose(np.asarray(phase.concentration(T, dmu)), gc, atol=1e-3)


def test_fast_interpolating_phase_inherits_concentration_range_xor():
    """Fast honours the SlowInterpolatingPhase concentration_range/extrapolation XOR (#278)."""
    phases = [
        LinePhase("A", fixed_concentration=0.2, line_energy=0.0),
        LinePhase("mid", fixed_concentration=0.5, line_energy=-0.3),
        LinePhase("B", fixed_concentration=0.8, line_energy=0.0),
    ]
    explicit = (0.1, 0.9)
    assert FastInterpolatingPhase("sol", phases, concentration_range=explicit).concentration_range == explicit
    with pytest.raises(ValueError, match="mutually exclusive"):
        FastInterpolatingPhase("sol", phases, concentration_range=explicit, maximum_extrapolation=0.05)


# --- LowTemperatureExpansionSublattice tests ---

_LTE_ATOL = 1e-12


def _lte_sublattice(eta=0.5):
    """A two-defect sublattice (antisite + vacancy) in the dilute limit."""
    antisite = ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1)
    vacancy = ConstantPointDefect("V_a", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0)
    return LowTemperatureExpansionSublattice(
        name="alpha", sublattice=0, sublattice_fraction=eta, defects=[antisite, vacancy]
    )


def _exact_sublattice(eta=0.5):
    """Same defects as :func:`_lte_sublattice`, exact site-competition model."""
    antisite = ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1)
    vacancy = ConstantPointDefect("V_a", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0)
    return PointDefectSublattice(
        name="alpha", sublattice=0, sublattice_fraction=eta, defects=[antisite, vacancy]
    )


def test_sublattice_models_are_siblings_under_abc():
    """Exact and LTE sublattices share the abstract base as siblings (not a
    subclass chain), and the base itself is abstract."""
    assert issubclass(PointDefectSublattice, AbstractPointDefectSublattice)
    assert issubclass(LowTemperatureExpansionSublattice, AbstractPointDefectSublattice)
    assert not issubclass(LowTemperatureExpansionSublattice, PointDefectSublattice)
    with pytest.raises(TypeError):
        AbstractPointDefectSublattice("a", 0, 0.5, [])


def test_lte_semigrand_potential_matches_leading_term():
    """phi_contrib = -eta * kB * T * sum_i exp(-([g_i] - [n_i] dmu) / kB / T)."""
    eta = 0.5
    sub = _lte_sublattice(eta)
    T = 600.0
    dmu = np.linspace(-0.4, 0.4, 17)
    z = np.array([np.exp(-(d.excess_free_energy(T) - d.excess_solutes * dmu) / kB / T) for d in sub.defects])
    expected = -eta * kB * T * z.sum(axis=0)
    assert_allclose(sub.semigrand_potential_contribution(T, dmu), expected, rtol=_LTE_ATOL, atol=0)


def test_lte_concentration_matches_leading_term():
    """c_contrib = eta * sum_i [n_i] exp(-([g_i] - [n_i] dmu) / kB / T) (no denominator)."""
    eta = 0.5
    sub = _lte_sublattice(eta)
    T = 600.0
    dmu = np.linspace(-0.4, 0.4, 17)
    z = np.array([np.exp(-(d.excess_free_energy(T) - d.excess_solutes * dmu) / kB / T) for d in sub.defects])
    nes = np.array([d.excess_solutes for d in sub.defects])
    expected = eta * (nes[:, None] * z).sum(axis=0)
    assert_allclose(sub.concentration_contribution(T, dmu), expected, rtol=_LTE_ATOL, atol=0)


def test_lte_concentration_is_minus_dphi_ddmu():
    """Gibbs-Duhem for the sublattice contribution: c = -d(phi)/d(dmu)."""
    sub = _lte_sublattice()
    T = 600.0
    dmu = np.linspace(-0.4, 0.4, 4000)
    phi = sub.semigrand_potential_contribution(T, dmu)
    c_numeric = -np.gradient(phi, dmu)
    c_direct = sub.concentration_contribution(T, dmu)
    # np.gradient is one-sided (less accurate) at the array ends
    assert_allclose(c_numeric[1:-1], c_direct[1:-1], rtol=1e-4, atol=1e-6)


def test_lte_converges_to_exact_in_dilute_limit():
    """At low T / large formation energies the defects are dilute and the
    leading expansion reproduces the exact site-competition model."""
    lte = _lte_sublattice()
    exact = _exact_sublattice()
    # kB*T ~ 0.026 eV and |dmu| <= 0.1 keep the smallest defect cost at 0.2 eV,
    # so the largest site fraction z ~ exp(-7.8) ~ 4e-4 and the dropped
    # 1/(1+sum z) correction is O(z).
    T = 300.0
    dmu = np.linspace(-0.1, 0.1, 21)
    assert_allclose(
        lte.semigrand_potential_contribution(T, dmu),
        exact.semigrand_potential_contribution(T, dmu),
        rtol=1e-3,
        atol=0,
    )
    assert_allclose(
        lte.concentration_contribution(T, dmu),
        exact.concentration_contribution(T, dmu),
        rtol=1e-3,
        atol=0,
    )


def test_lte_deviates_from_exact_when_not_dilute():
    """When defects are no longer dilute the leading expansion drops the
    site-competition denominator: |phi_lte| > |phi_exact| and the site
    fraction x_i = z_i is no longer bounded below one."""
    lte = _lte_sublattice()
    exact = _exact_sublattice()
    T = 4000.0  # kB*T ~ 0.34 eV, comparable to the formation energies
    dmu = 0.45  # drives the antisite population up
    phi_lte = lte.semigrand_potential_contribution(T, dmu)
    phi_exact = exact.semigrand_potential_contribution(T, dmu)
    # ln(1 + sum z) < sum z for z > 0, so the expansion is strictly more negative
    assert phi_lte < phi_exact - 1e-3
    # the leading-order site fraction exceeds the physical bound the exact model keeps
    z = sum(np.exp(-(d.excess_free_energy(T) - d.excess_solutes * dmu) / kB / T) for d in lte.defects)
    assert z > 1.0


def _lte_b2_phase():
    """B2 PointDefectedPhase built from two LTE sublattices."""
    host = LinePhase("AB", fixed_concentration=0.5, line_energy=-0.40, line_entropy=1.0 * kB)
    alpha = LowTemperatureExpansionSublattice(
        name="alpha",
        sublattice=0,
        sublattice_fraction=0.5,
        defects=[
            ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1),
            ConstantPointDefect("V_a", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
        ],
    )
    beta = LowTemperatureExpansionSublattice(
        name="beta",
        sublattice=1,
        sublattice_fraction=0.5,
        defects=[
            ConstantPointDefect("A_b", excess_energy=0.30, excess_entropy=0.0, excess_solutes=-1),
            ConstantPointDefect("V_b", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
        ],
    )
    return host, PointDefectedPhase(name="AB_lte", line_phase=host, sublattices=[alpha, beta])


def test_lte_phase_phi_c_consistency_full_range():
    """c = -d(phi)/d(dmu) holds for the LTE phase across the FULL mu range,
    *including* the saturated region. The concentration is clamped to [0, 1] and
    phi is continued as the tangent line past each saturation point, so the pair
    stays thermodynamically consistent where the dilute approximation breaks down
    (an independent c clamp without the phi continuation would desync them)."""
    _, phase = _lte_b2_phase()
    for T in (300.0, 800.0, 1500.0):
        dmu = np.linspace(-0.6, 0.6, 6000)
        phi = phase.semigrand_potential(T, dmu)
        c_numeric = -np.gradient(phi, dmu)
        c_direct = phase.concentration(T, dmu)
        # both saturated branches are genuinely exercised at the ends
        assert c_direct.max() == pytest.approx(1.0, abs=1e-9)
        assert c_direct.min() == pytest.approx(0.0, abs=1e-9)
        # ends dropped (np.gradient is one-sided there)
        assert_allclose(c_numeric[1:-1], c_direct[1:-1], rtol=5e-3, atol=1e-3)


def test_lte_phase_concentration_clamped_to_unit_interval():
    """Outside the dilute regime the unnormalised site fractions would leave
    [0, 1]; the phase clamps c (and tangent-continues phi to stay consistent)."""
    _, phase = _lte_b2_phase()
    dmu = np.linspace(-1.5, 1.5, 600)
    c = phase.concentration(2000.0, dmu)
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_lte_phase_phi_bounded_by_tangent_extension():
    """Raw phi = -kB*T*sum_i z_i plunges to -inf at large |dmu|; the tangent
    continuation past the saturation point keeps the phase's phi finite, which is
    what stops an out-of-range LTE phase from spuriously winning argmin-phi."""
    host, phase = _lte_b2_phase()
    T = 300.0
    raw_phi = host.semigrand_potential(T, 0.6) + sum(
        s.semigrand_potential_contribution(T, 0.6) for s in phase.sublattices
    )
    assert raw_phi < -100.0  # the divergence being guarded against
    assert phase.semigrand_potential(T, 0.6) > -2.0  # bounded by the tangent line


@settings(max_examples=75, deadline=None)
@given(
    e1=st.floats(0.3, 1.0),
    e2=st.floats(0.3, 1.0),
    s1=st.floats(0.0, 2e-4),
    n1=st.integers(-1, 1),
    n2=st.integers(-1, 1),
    eta=st.floats(0.1, 0.9),
    T=st.floats(400.0, 1500.0),
    dmu=st.floats(-0.2, 0.2),
)
def test_lte_contribution_phi_c_consistency_property(e1, e2, s1, n1, n2, eta, T, dmu):
    """Random LTE sublattice: concentration_contribution equals the numerical
    derivative -d/d(dmu) of semigrand_potential_contribution, so the two methods
    can never drift apart."""
    sub = LowTemperatureExpansionSublattice(
        name="s",
        sublattice=0,
        sublattice_fraction=eta,
        defects=[
            ConstantPointDefect("d1", excess_energy=e1, excess_entropy=s1, excess_solutes=n1),
            ConstantPointDefect("d2", excess_energy=e2, excess_entropy=0.0, excess_solutes=n2),
        ],
    )
    h = 1e-6
    fd = -(
        sub.semigrand_potential_contribution(T, dmu + h)
        - sub.semigrand_potential_contribution(T, dmu - h)
    ) / (2 * h)
    direct = sub.concentration_contribution(T, dmu)
    assert_allclose(direct, fd, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Back-compat re-exports of the pre-split point-defect API. The classes moved to
# landau.phases.pointdefects; the names below stay importable from landau.phases
# (concrete ones via a deprecation shim) until they are removed at the 2.0
# release -- delete this block together with the re-exports in
# landau/phases/__init__.py then.
# ---------------------------------------------------------------------------
_OLD_CONCRETE_REEXPORTS = ["ConstantPointDefect", "PointDefectSublattice", "PointDefectedPhase"]
_NEW_POINTDEFECT_CLASSES = ["AbstractPointDefectSublattice", "LowTemperatureExpansionSublattice"]


@pytest.mark.parametrize(
    "name", _OLD_CONCRETE_REEXPORTS + _NEW_POINTDEFECT_CLASSES + ["AbstractPointDefect"]
)
def test_pointdefect_classes_live_in_pointdefects_module(name):
    """All point-defect classes are importable from the canonical module."""
    import landau.phases.pointdefects as pdm

    assert isinstance(getattr(pdm, name), type)


@pytest.mark.parametrize("name", _OLD_CONCRETE_REEXPORTS + ["AbstractPointDefect"])
def test_old_pointdefect_api_still_importable_from_landau_phases(name):
    """The pre-split public names remain importable from landau.phases."""
    import landau.phases as phases

    assert hasattr(phases, name)


def test_old_pointdefect_abc_reexport_is_the_canonical_class():
    """AbstractPointDefect existed before the split and is re-exported as-is, so it
    stays usable as a base class (no deprecation shim)."""
    import landau.phases as phases
    import landau.phases.pointdefects as pdm

    assert phases.AbstractPointDefect is pdm.AbstractPointDefect


@pytest.mark.parametrize("name", _OLD_CONCRETE_REEXPORTS)
def test_old_concrete_reexport_is_deprecation_shim(name):
    """The concrete re-exports are deprecation wrappers, not the canonical classes."""
    import landau.phases as phases
    import landau.phases.pointdefects as pdm

    shim = getattr(phases, name)
    assert shim is not getattr(pdm, name)  # a wrapper, not the class itself
    assert hasattr(shim, "__wrapped__")  # functools.wraps marks the deprecation shim


def test_old_concrete_reexport_warns_and_constructs_canonical():
    """Constructing a concrete class via the deprecated landau.phases path warns and
    still yields an instance of the canonical class."""
    import landau.phases as phases
    import landau.phases.pointdefects as pdm

    with pytest.warns(DeprecationWarning):
        defect = phases.ConstantPointDefect("d", excess_energy=0.3, excess_entropy=0.0, excess_solutes=1)
    assert isinstance(defect, pdm.ConstantPointDefect)


@pytest.mark.parametrize("name", _NEW_POINTDEFECT_CLASSES)
def test_new_pointdefect_classes_not_added_to_landau_phases(name):
    """Classes introduced with the split have no back-compat obligation, so they
    are not re-exported from landau.phases (import from pointdefects instead)."""
    import landau.phases as phases

    assert not hasattr(phases, name)


# --- check_interpolation / check_concentration_interpolation plot_error ---

# residuals are recomputed with the same deterministic public methods the plot
# uses, so the plotted markers must match to numerical noise
ERR_ATOL = 1e-9


def _under_parametrised_interpolating_phase(add_entropy=False):
    """An InterpolatingPhase whose fit cannot pass through every sample, so the
    per-sample residuals are genuinely nonzero at the interior points."""
    cs = [0.0, 0.25, 0.5, 0.75, 1.0]
    fe = [0.0, -0.3, -0.1, -0.35, 0.05]
    lps = [LinePhase(n, c, f) for n, c, f in zip("ABCDE", cs, fe)]
    return InterpolatingPhase(name="sol", phases=lps, num_coeffs=3, add_entropy=add_entropy), lps


def test_check_interpolation_plot_error_plots_only_residuals():
    """plot_error=True draws interp(T) - sample at each sample and nothing else."""
    Ts = np.linspace(300, 1200, 40)
    Fs = -1e-3 * Ts**2 + 2.0  # nonlinear, so a 2-parameter (linear) fit leaves real residuals
    phase = TemperatureDependentLinePhase("L", 0.0, Ts, Fs, interpolator=PolyFit(2))

    fig, ax = plt.subplots()
    try:
        phase.check_interpolation(plot_error=True)
        assert len(ax.lines) == 0  # the free-energy curve is not drawn
        assert len(ax.collections) == 1  # only the error scatter
        n = max(int(len(Ts) // 100), 1)
        ts, fs = Ts[::n], Fs[::n]
        expected = phase.line_free_energy(ts) - fs
        # non-degenerate: a constant or all-zero residual plot would fail these
        assert np.abs(expected).max() > 1.0
        assert not np.allclose(expected, expected[0])
        off = np.asarray(ax.collections[0].get_offsets())
        assert_allclose(off[:, 0], ts)
        assert_allclose(off[:, 1], expected, atol=ERR_ATOL)
    finally:
        plt.close(fig)


def test_check_interpolation_default_plots_curve_and_samples():
    """Without plot_error the curve and its samples are drawn (the flag is opt-in)."""
    Ts = np.linspace(300, 1200, 20)
    phase = TemperatureDependentLinePhase("L", 0.0, Ts, np.sqrt(Ts), interpolator=PolyFit(2))
    fig, ax = plt.subplots()
    try:
        phase.check_interpolation()
        assert len(ax.lines) == 1  # the interpolation curve
        assert len(ax.collections) == 1  # the samples
    finally:
        plt.close(fig)


def test_check_concentration_interpolation_plot_error_plots_only_residuals():
    """plot_error=True draws free_energy(c) - sample at each line phase and nothing else."""
    phase, lps = _under_parametrised_interpolating_phase()
    fig, ax = plt.subplots()
    try:
        phase.check_concentration_interpolation(T=1000, plot_error=True)
        assert len(ax.lines) == 0  # the interpolation curve is not drawn
        cs = np.array([p.line_concentration for p in lps])
        expected = np.array([phase.free_energy(1000, p.line_concentration) - p.line_free_energy(1000) for p in lps])
        order = np.argsort(cs)
        assert np.abs(expected).max() > 1e-3  # the under-parametrised fit really misses the interior
        off = np.asarray(ax.collections[-1].get_offsets())
        off = off[np.argsort(off[:, 0])]
        assert_allclose(off[:, 0], cs[order])
        assert_allclose(off[:, 1], expected[order], atol=ERR_ATOL)
    finally:
        plt.close(fig)


def test_check_concentration_interpolation_error_invariant_to_plot_excess():
    """The excess shift moves curve and sample together, so the residuals are unchanged."""
    phase, _ = _under_parametrised_interpolating_phase()

    def errors(plot_excess):
        fig, ax = plt.subplots()
        try:
            phase.check_concentration_interpolation(T=1000, plot_excess=plot_excess, plot_error=True)
            off = np.asarray(ax.collections[-1].get_offsets())
            return off[np.argsort(off[:, 0])][:, 1]
        finally:
            plt.close(fig)

    assert_allclose(errors(False), errors(True), atol=ERR_ATOL)


def test_check_concentration_interpolation_error_respects_add_entropy():
    """With add_entropy the sample is the entropy-adjusted line value, against which the
    residual is measured."""
    phase, lps = _under_parametrised_interpolating_phase(add_entropy=True)
    fig, ax = plt.subplots()
    try:
        phase.check_concentration_interpolation(T=1000, plot_error=True)
        cs = np.array([p.line_concentration for p in lps])
        expected = np.array(
            [
                phase.free_energy(1000, p.line_concentration)
                - (p.line_free_energy(1000) - 1000 * S(p.line_concentration))
                for p in lps
            ]
        )
        order = np.argsort(cs)
        assert np.abs(expected).max() > 1e-3
        off = np.asarray(ax.collections[-1].get_offsets())
        off = off[np.argsort(off[:, 0])]
        assert_allclose(off[:, 1], expected[order], atol=ERR_ATOL)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("concentration_range", [(0.3, 0.7), (0.0, 1.0)])
@pytest.mark.parametrize("add_entropy", [False, True])
def test_check_concentration_interpolation_plot_excess_anchors_to_line_phases(
    concentration_range, add_entropy
):
    """plot_excess subtracts the chord between the *line phases'* free energies at
    their own (min/max) concentrations -- not at the concentration_range bounds.
    The two coincide only when concentration_range matches the line phases' span;
    here it doesn't, both narrower (0.3, 0.7) and wider (0.0, 1.0) than the
    phases' [0.2, 0.8], so the extremal line phases must land exactly on zero in
    both directions and under add_entropy (which exercises the S(c) correction
    at the line phase's own concentration rather than the range bound)."""
    cs = [0.2, 0.5, 0.8]
    fe = [0.0, -0.3, 0.05]
    lps = [LinePhase(n, c, f) for n, c, f in zip("ABC", cs, fe)]
    phase = SlowInterpolatingPhase(
        name="sol",
        phases=lps,
        concentration_range=concentration_range,
        interpolator=PolyFit(2),
        add_entropy=add_entropy,
    )

    fig, ax = plt.subplots()
    try:
        phase.check_concentration_interpolation(T=1000, plot_excess=True)
        offsets = {tuple(c.get_offsets()[0]) for c in ax.collections}
        anchors = {c for c in offsets if c[0] in (0.2, 0.8)}
        assert len(anchors) == 2
        for _, y in anchors:
            assert abs(y) < ERR_ATOL
    finally:
        plt.close(fig)
