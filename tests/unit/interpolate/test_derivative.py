"""Direct unit tests for NumericalDerivative, the default Interpolation.deriv().

Covers the vectorised central-difference fallback used by closure-wrapped
fits (SplineFit, StitchedFit, softplus, Whitney) that have no closed-form
derivative.
"""

import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from landau.interpolate.basic import NumericalDerivative


def _constant(x):
    return np.full_like(np.asarray(x, dtype=float), 5.0)


def _linear(x):
    return 2 * x + 3


def _quadratic(x):
    return x**2


def test_constant_derivative_is_zero():
    d = NumericalDerivative(_constant)
    x = np.linspace(-3, 3, 7)
    assert np.allclose(d(x), 0.0, atol=1e-8)


def test_linear_derivative_matches_slope():
    d = NumericalDerivative(_linear)
    assert np.isclose(d(0.5), 2.0)
    x = np.linspace(-5, 5, 11)
    assert np.allclose(d(x), 2.0)


def test_quadratic_derivative_matches_analytic():
    d = NumericalDerivative(_quadratic)
    x = np.linspace(1, 4, 6)  # interior, away from x=0 where relative step degenerates
    assert np.allclose(d(x), 2 * x, atol=1e-4)


def test_scalar_and_array_shape_contract():
    d = NumericalDerivative(_quadratic)

    scalar_out = d(0.5)
    assert np.ndim(scalar_out) == 0

    x = np.linspace(1, 4, 6)
    array_out = d(x)
    assert array_out.shape == x.shape


@given(
    coeffs=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=6),  # order < 6, i.e. degree <= 5
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    )
)
def test_derivative_matches_random_polynomial(coeffs):
    def poly(x):
        return np.polyval(coeffs[::-1], x)

    d = NumericalDerivative(poly)
    x = np.linspace(1, 4, 9)  # interior, away from x=0 where relative step degenerates
    analytic = np.polyval(np.polyder(coeffs[::-1]), x)
    assert np.allclose(d(x), analytic, atol=1e-4)
