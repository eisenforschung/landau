import numpy as np
import pytest
from landau.interpolate import RedlichKister, RedlichKisterInterpolation
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

EXACT_ATOL = 1e-12
"""The fitted parameters against the least-squares ones the samples were built from (measured 3.1e-14)."""

def test_RedlichKister_terminal_check():
    rk = RedlichKister(nparam=2)
    c_no_term = np.linspace(0.1, 0.9, 10)
    y_no_term = np.zeros_like(c_no_term)
    with pytest.raises(AssertionError, match="Must include terminals when fitting Redlich-Kister!"):
        rk.fit(c_no_term, y_no_term)

@given(
    L=arrays(dtype=float, shape=st.integers(min_value=1, max_value=5), elements=st.floats(min_value=-1, max_value=1)),
    f0=st.floats(min_value=-1, max_value=1),
    df=st.floats(min_value=-1, max_value=1),
    noise=arrays(dtype=float, shape=20, elements=st.floats(min_value=-1, max_value=1)),
)
def test_RedlichKister_fit_is_the_least_squares_solution(L, f0, df, noise):
    """Samples of a Redlich-Kister curve plus a residual orthogonal to the fit's basis and zero at
    the terminals: the least-squares parameters are the curve's own, whatever the residual."""
    c = np.linspace(0, 1, 20)
    basis = (c * (1 - c))[:, None] * np.vander(2 * c - 1, len(L), increasing=True)
    q, _ = np.linalg.qr(basis)
    residual = noise.copy()
    residual[[0, -1]] = 0
    residual -= q @ (q.T @ residual)
    fit = RedlichKister(len(L)).fit(c, RedlichKisterInterpolation(df, f0, L)(c) + residual)
    np.testing.assert_allclose(fit.rk_parameters, L, rtol=0, atol=EXACT_ATOL)
    assert fit.f0 == pytest.approx(f0, abs=EXACT_ATOL)
    assert fit.df == pytest.approx(df, abs=EXACT_ATOL)


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
