import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import Boltzmann, eV

from landau.interpolate import SGTE
from landau.interpolate.basic import G_calphad
from landau.phases import LinePhase, TemperatureDependentLinePhase, IdealSolution

kB = Boltzmann / eV


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


def _make_ideal_solution(f1=-1.0, f2=-2.0):
    p1 = LinePhase(name="A", fixed_concentration=0, line_energy=f1)
    p2 = LinePhase(name="B", fixed_concentration=1, line_energy=f2)
    return IdealSolution(name="AB", phase1=p1, phase2=p2), f1, f2


def test_ideal_solution_semigrand_at_dmu_eq_df():
    sol, f1, f2 = _make_ideal_solution()
    T = 500.0
    df = f2 - f1
    phi = sol.semigrand_potential(T, df)
    assert_allclose(phi, f1 - kB * T * np.log(2), rtol=1e-10)


def test_ideal_solution_semigrand_large_positive_dmu():
    # dmu >> df triggers the non-finite overflow branch; phi -> f2 - dmu
    sol, f1, f2 = _make_ideal_solution()
    T = 500.0
    dmu = 100.0
    phi = sol.semigrand_potential(T, dmu)
    assert_allclose(phi, f2 - dmu, rtol=1e-6)


def test_ideal_solution_semigrand_large_negative_dmu():
    # dmu << 0: exp term vanishes; phi -> f1
    sol, f1, f2 = _make_ideal_solution()
    T = 500.0
    dmu = -100.0
    phi = sol.semigrand_potential(T, dmu)
    assert_allclose(phi, f1, rtol=1e-6)


def test_ideal_solution_concentration_at_dmu_eq_df():
    sol, f1, f2 = _make_ideal_solution()
    T = 500.0
    df = f2 - f1
    c = sol.concentration(T, df)
    assert_allclose(c, 0.5, rtol=1e-10)
