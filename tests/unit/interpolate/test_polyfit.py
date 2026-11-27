import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from landau.interpolate import PolyFit, basic

CURVATURE_ATOL = 1e-8

def test_PolyFit_fixed():
    x = np.linspace(0, 1, 10)
    y = 2 * x**2 + 3 * x + 1
    pf = PolyFit(nparam=3)
    fit = pf.fit(x, y)
    assert np.allclose(fit(x), y)

def test_PolyFit_auto():
    x = np.linspace(0, 1, 10)
    y = 2 * x**2 + 3 * x + 1
    pf_auto = PolyFit(nparam="auto")
    fit_auto = pf_auto.fit(x, y)
    assert np.allclose(fit_auto(x), y, atol=1e-5)

def _convex_samples():
    """Convex data: the unconstrained fit reproduces it, a concave one cannot."""
    x = np.linspace(0, 1, 15)
    return x, 2 * x**2 - x + 0.5

@pytest.mark.skipif(basic.polyfit is None, reason="needs the constraints extra (polyfit)")
def test_PolyFit_curvature_constrains_the_fit():
    x, y = _convex_samples()
    fit = PolyFit(nparam=3, enforce_curvature=True).fit(x, y)
    assert fit.deriv().deriv()(x).max() <= CURVATURE_ATOL
    # The best concave fit to convex data is the least-squares line, so the fit
    # still follows the data's trend; a constant would fail this.
    assert np.allclose(fit(x), np.polyval(np.polyfit(x, y, 1), x), atol=CURVATURE_ATOL)

def test_PolyFit_curvature_warns_without_polyfit(monkeypatch):
    monkeypatch.setattr(basic, "polyfit", None)
    x, y = _convex_samples()
    with pytest.warns(UserWarning, match="enforce_curvature=True is only supported when the `polyfit` package"):
        fit = PolyFit(nparam=3, enforce_curvature=True).fit(x, y)
    # The fallback is the unconstrained fit, which keeps the data's convexity.
    assert np.array_equal(fit.coefficients, PolyFit(nparam=3).fit(x, y).coefficients)
    assert np.allclose(fit(x), y, atol=1e-5)

@given(
    coeffs=arrays(dtype=float, shape=st.integers(min_value=1, max_value=5), elements=st.floats(min_value=-10, max_value=10))
)
def test_PolyFit_hypothesis(coeffs):
    x = np.linspace(0, 1, 20)
    y = np.polyval(coeffs[::-1], x)
    pf = PolyFit(nparam=len(coeffs))
    fit = pf.fit(x, y)
    assert np.allclose(fit(x), y, atol=1e-5)
    # Check that coefficients are reproduced, via the interpolation's own
    # ascending-power interface rather than the wrapped poly1d's descending
    # order.  Ridge regularization biases the recovered coefficients; the bias
    # grows with polynomial degree and reaches ~1.5e-3 for the worst-conditioned
    # degree-5 case on x in [0, 1], so allow a tolerance with margin above that.
    # The fit values themselves are recovered far more tightly (atol=1e-5 above).
    assert np.allclose(fit.coefficients, coeffs, atol=5e-3)
