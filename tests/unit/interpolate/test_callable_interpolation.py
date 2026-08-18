"""Direct unit tests for _CallableInterpolation, the closure adapter that wraps
every non-analytic Interpolation (spline / stitched / softplus / Whitney, and
the derivative object returned by SGTEInterpolation.deriv()).

Covers its two responsibilities: forwarding __call__ verbatim to the wrapped
callable, and inheriting the numerical Interpolation.deriv() fallback.
"""

import numpy as np

from landau.interpolate.basic import NumericalDerivative, _CallableInterpolation


def test_call_forwards_scalar_input_verbatim():
    wrapped = _CallableInterpolation(lambda x: 2 * x + 3)
    assert wrapped(0.5) == 4.0


def test_call_forwards_array_input_verbatim():
    x = np.array([0.0, 1.0, 2.0])
    wrapped = _CallableInterpolation(lambda x: x**2)
    assert np.array_equal(wrapped(x), x**2)


def test_deriv_returns_numerical_derivative():
    wrapped = _CallableInterpolation(lambda x: x**2)
    d = wrapped.deriv()
    assert isinstance(d, NumericalDerivative)


def test_deriv_recovers_analytic_derivative():
    # d/dx (x**2) = 2*x
    wrapped = _CallableInterpolation(lambda x: x**2)
    x = np.linspace(1, 4, 6)  # interior, away from x=0 where relative step degenerates
    assert np.allclose(wrapped.deriv()(x), 2 * x, atol=1e-4)
