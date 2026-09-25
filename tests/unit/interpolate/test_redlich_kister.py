from fractions import Fraction

import numpy as np
import pytest
from landau.interpolate import RedlichKister, RedlichKisterInterpolation
from landau.phases import S, kB
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

EXACT_ATOL = 1e-10
"""eV; the fitted curve against the least-squares solution in exact arithmetic (measured 1.1e-12)."""

def test_RedlichKister_terminal_check():
    rk = RedlichKister(nparam=2)
    c_no_term = np.linspace(0.1, 0.9, 10)
    y_no_term = np.zeros_like(c_no_term)
    with pytest.raises(AssertionError, match="Must include terminals when fitting Redlich-Kister!"):
        rk.fit(c_no_term, y_no_term)

@given(
    L=arrays(dtype=float, shape=st.integers(min_value=1, max_value=3), elements=st.floats(min_value=-1, max_value=1)),
    f0=st.floats(min_value=-1, max_value=1),
    df=st.floats(min_value=-1, max_value=1)
)
def test_RedlichKister_hypothesis(L, f0, df):
    c = np.linspace(0, 1, 20)
    y_mix = RedlichKisterInterpolation._eval_mix(c, *L)
    y = y_mix + f0 + df * c
    rk = RedlichKister(nparam=len(L))
    fit = rk.fit(c, y)
    assert np.allclose(fit(c), y, atol=1e-5)
    assert np.isclose(fit.f0, f0, atol=1e-5)
    assert np.isclose(fit.df, df, atol=1e-5)
    assert np.allclose(fit.rk_parameters, L, atol=1e-5)


def _exact_least_squares(c, f, nparam):
    """``(df, f0, L)``, in the order ``RedlichKisterInterpolation`` takes them, solved from
    the normal equations in exact rationals: the float inputs are exact binary fractions,
    so nothing in this reference is rounded."""
    c = [Fraction(x) for x in c]
    f = [Fraction(y) for y in f]
    f0, df = f[0], f[-1] - f[0]
    r = [y - f0 - df * x for x, y in zip(c, f)]
    B = [[x * (1 - x) * (2 * x - 1) ** v for v in range(nparam)] for x in c]
    A = [[sum(row[i] * row[j] for row in B) for j in range(nparam)] for i in range(nparam)]
    b = [sum(row[i] * ri for row, ri in zip(B, r)) for i in range(nparam)]
    for i in range(nparam):
        for k in range(i + 1, nparam):
            factor = A[k][i] / A[i][i]
            A[k] = [a - factor * p for a, p in zip(A[k], A[i])]
            b[k] -= factor * b[i]
    L = [Fraction(0)] * nparam
    for i in reversed(range(nparam)):
        L[i] = (b[i] - sum(A[i][j] * L[j] for j in range(i + 1, nparam))) / A[i][i]
    return float(df), float(f0), np.array([float(v) for v in L])


def test_RedlichKister_fit_is_the_least_squares_solution():
    """Eight line phases at 1000 K, their free energies not a Redlich-Kister curve,
    five parameters: the fit is the least-squares solution, not an approximation of it."""
    c = np.array([0.0, 0.595, 0.619, 0.697, 0.751, 0.808, 0.888, 1.0])
    energy = np.array([-2.815, -3.108, -3.137, -2.994, -2.857, -2.897, -3.531, -2.857])
    entropy = np.array([0.873, 1.536, 0.705, 2.099, 2.529, 1.183, 1.826, 0.956]) * kB
    f = energy - 1000.0 * entropy + 1000.0 * S(c)
    fit = RedlichKister(5).fit(c, f.copy())
    exact = RedlichKisterInterpolation(*_exact_least_squares(c, f, 5))
    grid = np.linspace(0, 1, 201)
    np.testing.assert_allclose(fit(grid), exact(grid), rtol=0, atol=EXACT_ATOL)


def test_RedlichKister_fit_through_the_terminals_alone_is_the_chord():
    c = np.array([0.0, 1.0])
    f = np.array([-3.0, -2.5])
    fit = RedlichKister(3).fit(c, f.copy())
    assert fit.rk_parameters.shape == (0,)
    grid = np.linspace(0, 1, 11)
    np.testing.assert_allclose(fit(grid), -3.0 + 0.5 * grid, rtol=0, atol=1e-15)


def test_RedlichKister_fit_that_is_not_unique_is_the_minimum_norm_solution():
    """Two parameters through one interior concentration: the fit passes through the mean of
    the two samples there, and its parameters have no component along the free direction."""
    c = np.array([0.0, 0.3, 0.3, 1.0])
    f = np.array([-3.0, -3.2, -3.1, -2.5])
    fit = RedlichKister(2).fit(c, f.copy())
    assert fit(0.3) == pytest.approx(-3.15, abs=1e-14)
    # the basis row at c=0.3 is 0.21 * (1, 2*0.3 - 1); the free direction is orthogonal to it
    free = np.array([0.4, 1.0])
    assert fit.rk_parameters @ free == pytest.approx(0, abs=1e-14)


def test_eval_mix_derivative_scalar():
    """_eval_mix_derivative at scalar x should match finite-difference gradient of _eval_mix."""
    L = np.array([0.5, -0.3])
    x = 0.4
    eps = 1e-6
    fd = (RedlichKisterInterpolation._eval_mix(x + eps, *L) - RedlichKisterInterpolation._eval_mix(x - eps, *L)) / (2 * eps)
    deriv = RedlichKisterInterpolation._eval_mix_derivative(x, *L)
    assert np.isclose(deriv, fd, rtol=1e-4)


def test_eval_mix_derivative_array():
    """_eval_mix_derivative on an array should match element-wise finite differences."""
    L = np.array([0.8, -0.2, 0.1])
    c = np.linspace(0.1, 0.9, 15)
    eps = 1e-6
    fd = (RedlichKisterInterpolation._eval_mix(c + eps, *L) - RedlichKisterInterpolation._eval_mix(c - eps, *L)) / (2 * eps)
    deriv = RedlichKisterInterpolation._eval_mix_derivative(c, *L)
    assert np.allclose(deriv, fd, rtol=1e-4)


@given(
    L=arrays(dtype=float, shape=st.integers(min_value=1, max_value=3), elements=st.floats(min_value=-1, max_value=1)),
    x=st.floats(min_value=0.05, max_value=0.95)
)
def test_eval_mix_derivative_hypothesis(L, x):
    """_eval_mix_derivative should agree with finite differences for any L and interior x."""
    eps = 1e-6
    fd = (RedlichKisterInterpolation._eval_mix(x + eps, *L) - RedlichKisterInterpolation._eval_mix(x - eps, *L)) / (2 * eps)
    deriv = RedlichKisterInterpolation._eval_mix_derivative(x, *L)
    assert np.isclose(deriv, fd, rtol=1e-3, atol=1e-6)
