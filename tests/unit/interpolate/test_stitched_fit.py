import pytest
import numpy as np
from landau.interpolate import StitchedFit, PolyFit, SGTE, G_calphad
from hypothesis import given, strategies as st

@given(
    t_min=st.floats(min_value=51.0, max_value=500.0),
    t_max=st.floats(min_value=1001.0, max_value=2000.0),
    a=st.floats(min_value=-1e-6, max_value=1e-6),
    b=st.floats(min_value=-1, max_value=1),
    c=st.floats(min_value=-100, max_value=100),
    edge=st.integers(min_value=10, max_value=20)
)
def test_stitched_fit_properties(t_min, t_max, a, b, c, edge):
    def func(x):
        return a * x**2 + b * x + c

    t = np.linspace(t_min, t_max, 100)
    y = func(t)

    # Use quadratic for both to ensure better matching at boundaries
    stitched = StitchedFit(
        interpolating=PolyFit(nparam=3),
        low=PolyFit(nparam=3),
        upp=PolyFit(nparam=3),
        edge=edge
    )
    fit = stitched.fit(t, y)

    # 1. it fits the original data well in the interpolating region
    assert np.allclose(fit(t), y, atol=1e-2)

    # 2. low/upp fit the edge region reasonably
    t_low_extrap = np.linspace(t_min - 10, t_min, 5)
    t_upp_extrap = np.linspace(t_max, t_max + 10, 5)
    assert np.allclose(fit(t_low_extrap), func(t_low_extrap), atol=1e-1)
    assert np.allclose(fit(t_upp_extrap), func(t_upp_extrap), atol=1e-1)

    # 3. around the transition to low or upp the combined function is at least roughly continuous to the first derivative
    dt = 1e-2 # Use larger dt to avoid numerical issues with jumps
    # Check value continuity
    assert np.isclose(fit(t_min - dt), fit(t_min + dt), atol=1e-1)
    assert np.isclose(fit(t_max - dt), fit(t_max + dt), atol=1e-1)

    # Check gradient smoothness by comparing model slopes on both sides of boundaries
    # We use fit points strictly within their respective model regions to estimate slopes
    grad_low = (fit(t_min - dt) - fit(t_min - 2*dt)) / dt
    grad_mid_low = (fit(t_min + 2*dt) - fit(t_min + dt)) / dt
    assert np.isclose(grad_low, grad_mid_low, atol=1e-1)

    grad_mid_upp = (fit(t_max - dt) - fit(t_max - 2*dt)) / dt
    grad_upp = (fit(t_max + 2*dt) - fit(t_max + dt)) / dt
    assert np.isclose(grad_mid_upp, grad_upp, atol=1e-1)


def test_stitched_fit_sgte_mid_accuracy():
    """SGTE(2) fits T*log(T) exactly; StitchedFit mid-region should recover the data to high precision."""
    T = np.linspace(300, 1000, 50)
    y = G_calphad(T, 1.0, 0.0)  # = T*log(T)
    sf = StitchedFit(interpolating=SGTE(2), low=None, upp=None)
    fit = sf.fit(T, y)
    np.testing.assert_allclose(fit(T), y, rtol=1e-3)


def test_stitched_fit_scalar_input():
    """StitchedFit should handle scalar input without error."""
    T = np.linspace(300, 1000, 50)
    y = T * np.log(T)
    sf = StitchedFit(interpolating=PolyFit(nparam=3), low=None, upp=PolyFit(2))
    fit = sf.fit(T, y)
    result = fit(500.0)
    assert np.isfinite(result)


def test_stitched_fit_low_branch():
    """When low= is set, temperatures below tmin should use the low interpolator."""
    T = np.linspace(300, 1000, 50)
    y = 0.001 * T**2 - T + 10.0
    sf = StitchedFit(interpolating=PolyFit(nparam=3), low=PolyFit(nparam=2), upp=None, edge=10)
    fit = sf.fit(T, y)
    # Values inside the range should match closely
    np.testing.assert_allclose(fit(T), y, atol=0.1)
    # Value just below tmin should be finite (served by low branch)
    assert np.isfinite(fit(299.0))


# shared tolerance for the exact-recovery assertions below; all of them fit a polynomial through
# points that lie exactly on it, so the only error is PolyFit's ridge regularizer (~1e-7 relative)
_ATOL = 1e-4


def _quadratic(t):
    return 1e-4 * t**2 - 0.3 * t + 5.0


def test_stitched_fit_is_order_independent():
    """Shuffling (t, f) must not move the fitted curve, inside or outside the window."""
    t = np.linspace(300, 1000, 40)
    y = _quadratic(t)
    sf = StitchedFit(interpolating=PolyFit(3), low=PolyFit(2), upp=PolyFit(2), edge=10)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(t))

    probe = np.array([250.0, 500.0, 1100.0])
    np.testing.assert_allclose(sf.fit(t[perm], y[perm])(probe), sf.fit(t, y)(probe), atol=_ATOL)


def test_stitched_fit_edges_fitted_from_the_extreme_samples():
    """The edge fits see only the extreme samples, so they recover what the mid fit cannot."""
    t = np.linspace(300, 1000, 40)
    y = _quadratic(t)
    # linear mid fit cannot represent the quadratic, quadratic edge fits recover it exactly
    sf = StitchedFit(interpolating=PolyFit(2), low=PolyFit(3), upp=PolyFit(3), edge=10)
    fit = sf.fit(t, y)

    outside = np.array([250.0, 1100.0])
    np.testing.assert_allclose(fit(outside), _quadratic(outside), atol=_ATOL)
    # the mid fit is off by >10 units there, so neither it nor a wider edge window passes the above
    assert np.abs(PolyFit(2).fit(t, y)(outside) - _quadratic(outside)).min() > 10.0


def test_stitched_fit_edge_clamped_to_half_the_samples():
    """An edge wider than half the data is clamped, so the two windows stay disjoint."""
    t = np.arange(8.0)
    # lower half on one line, upper half on another
    y = np.where(t < 4, 2.0 * t, 10.0 - t)
    sf = StitchedFit(interpolating=PolyFit(3), low=PolyFit(2), upp=PolyFit(2), edge=100)
    fit = sf.fit(t, y)

    assert fit(-1.0) == pytest.approx(-2.0, abs=_ATOL)  # 2 * t continued below t = 0
    assert fit(8.0) == pytest.approx(2.0, abs=_ATOL)  # 10 - t continued above t = 7


def test_stitched_fit_single_sample():
    """A one-sample fit keeps one edge sample rather than an empty window."""
    sf = StitchedFit(interpolating=PolyFit(1), low=PolyFit(1), upp=PolyFit(1), edge=10)
    fit = sf.fit(np.array([500.0]), np.array([3.0]))

    np.testing.assert_allclose(fit(np.array([400.0, 500.0, 600.0])), 3.0, atol=_ATOL)
