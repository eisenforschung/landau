"""
Tests for QuasiChemicalPhase (quasi-chemical / pair approximation).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import Boltzmann, eV

from landau import QuasiChemicalPhase, LinePhase

kB = Boltzmann / eV


def _make_symmetric(V=0.0, z=12):
    """Symmetric binary with equal terminal energies, convenient for testing."""
    phase_A = LinePhase("A", fixed_concentration=0, line_energy=-1.0)
    phase_B = LinePhase("B", fixed_concentration=1, line_energy=-1.0)
    return QuasiChemicalPhase("test", phase_A=phase_A, phase_B=phase_B, V=V, z=z)


# --- construction -----------------------------------------------------------


def test_terminal_order_normalised():
    """Phase accepts terminals in any order and stores them canonically."""
    phase_A = LinePhase("A", fixed_concentration=0, line_energy=-1.0)
    phase_B = LinePhase("B", fixed_concentration=1, line_energy=-1.0)
    p1 = QuasiChemicalPhase("p", phase_A=phase_A, phase_B=phase_B, V=0.0)
    p2 = QuasiChemicalPhase("p", phase_A=phase_B, phase_B=phase_A, V=0.0)
    assert p1.phase_A.line_concentration == 0
    assert p2.phase_A.line_concentration == 0


def test_bad_terminals_raise():
    non_terminal = LinePhase("X", fixed_concentration=0.5, line_energy=0.0)
    terminal = LinePhase("B", fixed_concentration=1, line_energy=0.0)
    with pytest.raises(ValueError):
        QuasiChemicalPhase("bad", phase_A=non_terminal, phase_B=terminal, V=0.0)


# --- free energy limits ------------------------------------------------------


def test_free_energy_terminals():
    """Free energy at pure endpoints equals terminal free energies."""
    T = 1000.0
    phase = _make_symmetric(V=0.05)
    assert_allclose(phase.free_energy(T, 0.0), -1.0, rtol=1e-10)
    assert_allclose(phase.free_energy(T, 1.0), -1.0, rtol=1e-10)


def test_free_energy_symmetric():
    """Symmetric alloy: F(c) == F(1-c)."""
    T = 800.0
    phase = _make_symmetric(V=0.03)
    c = np.linspace(0.05, 0.95, 9)
    f = phase.free_energy(T, c)
    assert_allclose(f, phase.free_energy(T, 1.0 - c), rtol=1e-6)


def test_free_energy_array_T():
    """free_energy broadcasts correctly over T and c arrays."""
    phase = _make_symmetric(V=0.02)
    T = np.array([500.0, 1000.0, 1500.0])
    c = np.linspace(0.1, 0.9, 4)
    T_grid = T[:, np.newaxis]
    c_grid = c[np.newaxis, :]
    f = phase.free_energy(T_grid, c_grid)
    assert f.shape == (3, 4)


# --- pair fraction -----------------------------------------------------------


def test_y_AB_V0_symmetric():
    """At V=0, c=0.5: QCA pair equilibrium gives y_AB = 1/3 (Bethe-lattice result)."""
    phase = _make_symmetric(V=0.0, z=12)
    y = phase._y_AB(0.5, 1000.0)
    assert_allclose(y, 1.0 / 3.0, rtol=1e-5)


def test_y_AB_increases_with_V():
    """Positive V (ordering) increases y_AB compared to V=0."""
    T = 1000.0
    c = 0.5
    y0 = _make_symmetric(V=0.0)._y_AB(c, T)
    yp = _make_symmetric(V=0.05)._y_AB(c, T)
    assert yp > y0


def test_y_AB_decreases_with_negative_V():
    """Negative V (segregation) decreases y_AB compared to V=0."""
    T = 1000.0
    c = 0.5
    y0 = _make_symmetric(V=0.0)._y_AB(c, T)
    yn = _make_symmetric(V=-0.05)._y_AB(c, T)
    assert yn < y0


def test_y_AB_conservation():
    """Pair fractions satisfy y_AA = (1-c) - y_AB/2 and y_BB = c - y_AB/2 >= 0."""
    phase = _make_symmetric(V=0.04)
    c = np.linspace(0.01, 0.99, 20)
    T = 900.0
    y_AB = phase._y_AB(c, T)
    y_AA = (1.0 - c) - y_AB / 2.0
    y_BB = c - y_AB / 2.0
    assert np.all(y_AA >= -1e-12)
    assert np.all(y_BB >= -1e-12)
    assert_allclose(y_AA + y_AB + y_BB, 1.0, atol=1e-10)


def test_y_AB_equilibrium_condition():
    """Equilibrium pair fractions satisfy y_AB^2 / (y_AA * y_BB) = exp(V / kB / T)."""
    T = 1200.0
    V = 0.03
    c = 0.4
    phase = _make_symmetric(V=V)
    y_AB = phase._y_AB(c, T)
    y_AA = (1.0 - c) - y_AB / 2.0
    y_BB = c - y_AB / 2.0
    expected_ratio = np.exp(V / (kB * T))
    assert_allclose(y_AB**2 / (y_AA * y_BB), expected_ratio, rtol=1e-6)


# --- concentration symmetry and monotonicity ---------------------------------


def test_concentration_symmetric_at_zero_dmu():
    """Symmetric alloy at dmu=0 has c=0.5."""
    phase = _make_symmetric(V=0.04)
    c = phase.concentration(1000.0, 0.0)
    assert_allclose(c, 0.5, atol=1e-3)


def test_concentration_increases_with_dmu():
    """Concentration is a non-decreasing function of dmu."""
    phase = _make_symmetric(V=0.04)
    dmu = np.linspace(-0.5, 0.5, 20)
    c = phase.concentration(1000.0, dmu)
    assert np.all(np.diff(c) >= -1e-6)


def test_semigrand_decreases_with_T():
    """Semigrand potential at dmu=0 decreases as T rises (entropy lowers free energy)."""
    phase = _make_symmetric(V=0.03)
    dmu = 0.0
    phi_low = phase.semigrand_potential(500.0, dmu)
    phi_high = phase.semigrand_potential(2000.0, dmu)
    assert phi_high < phi_low


# --- FCC vs BCC --------------------------------------------------------------


def test_coordination_number_matters():
    """FCC (z=12) and BCC (z=8) give different free energies for the same V."""
    T = 1000.0
    c = 0.5
    p_fcc = _make_symmetric(V=0.05, z=12)
    p_bcc = _make_symmetric(V=0.05, z=8)
    assert not np.isclose(p_fcc.free_energy(T, c), p_bcc.free_energy(T, c))


# --- scalar / array API ------------------------------------------------------


def test_semigrand_scalar_returns_scalar():
    phase = _make_symmetric()
    result = phase.semigrand_potential(1000.0, 0.0)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)


def test_concentration_scalar_returns_scalar():
    phase = _make_symmetric()
    result = phase.concentration(1000.0, 0.0)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)


def test_semigrand_array_shape():
    phase = _make_symmetric()
    dmu = np.linspace(-0.3, 0.3, 5)
    phi = phase.semigrand_potential(1000.0, dmu)
    assert phi.shape == (5,)
