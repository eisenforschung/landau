"""Direct unit tests for PolynomialInterpolation, the analytic-derivative
Interpolation subclass returned by PolyFit.fit.

Covers the three responsibilities the class adds on top of the wrapped
numpy.poly1d: the ascending-power coefficient order, the scalar/array
shape contract inherited from the Interpolation protocol, and the
analytic .deriv() that FastInterpolatingPhase's Newton polish relies on.
"""

import numpy as np

from landau.interpolate.basic import PolynomialInterpolation


def test_coefficients_are_ascending_power_order():
    # np.poly1d takes descending powers: [2, 3, 1] means 2*x**2 + 3*x + 1
    poly = np.poly1d([2, 3, 1])
    p = PolynomialInterpolation(poly)
    # .coefficients decouples callers from poly1d's descending internal order
    assert np.array_equal(p.coefficients, [1, 3, 2])


def test_call_scalar_and_array_shape_contract():
    p = PolynomialInterpolation(np.poly1d([2, 3, 1]))
    # scalar in → scalar out (ndim == 0)
    scalar_out = p(0.5)
    assert np.ndim(scalar_out) == 0
    assert np.isclose(scalar_out, 2 * 0.25 + 3 * 0.5 + 1)
    # array in → array out with matching shape
    x = np.array([0.0, 1.0, 2.0])
    array_out = p(x)
    assert array_out.shape == x.shape
    assert np.array_equal(array_out, [1, 6, 15])


def test_deriv_returns_polynomial_interpolation():
    p = PolynomialInterpolation(np.poly1d([2, 3, 1]))
    d = p.deriv()
    # analytic derivative — not the default NumericalDerivative fallback that
    # FastInterpolatingPhase would silently accept but converge worse under
    assert isinstance(d, PolynomialInterpolation)


def test_deriv_coefficients_match_analytic_derivative():
    # d/dx (2 x^2 + 3 x + 1) = 4 x + 3
    p = PolynomialInterpolation(np.poly1d([2, 3, 1]))
    assert np.array_equal(p.deriv().coefficients, [3, 4])
    # constant polynomial differentiates to 0
    constant = PolynomialInterpolation(np.poly1d([7]))
    assert np.array_equal(constant.deriv().coefficients, [0])
