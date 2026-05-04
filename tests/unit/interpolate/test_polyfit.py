import numpy as np
import warnings
import importlib.util
from landau.interpolate import PolyFit
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

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

def test_PolyFit_curvature():
    x = np.linspace(0, 1, 10)
    # Use concave data as Gibbs energy is concave
    y_concave = -2 * x**2 + 3 * x + 1
    pf_curve = PolyFit(nparam=3, enforce_curvature=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_curve = pf_curve.fit(x, y_concave)

        if importlib.util.find_spec("polyfit") is None:
            # If polyfit is missing, it should warn
            assert any("enforce_curvature=True is only supported when the `polyfit` package" in str(warn.message) for warn in w)

        # It should still have produced a valid fit
        assert np.allclose(fit_curve(x), y_concave, atol=1e-5)

@given(
    coeffs=arrays(dtype=float, shape=st.integers(min_value=1, max_value=5), elements=st.floats(min_value=-10, max_value=10))
)
def test_PolyFit_hypothesis(coeffs):
    x = np.linspace(0, 1, 20)
    y = np.polyval(coeffs[::-1], x)
    pf = PolyFit(nparam=len(coeffs))
    fit = pf.fit(x, y)
    assert np.allclose(fit(x), y, atol=1e-5)
    # Check that coefficients are reproduced
    # Use looser tolerance due to Ridge regularization
    assert np.allclose(fit.coeffs, coeffs[::-1], atol=1e-3)
