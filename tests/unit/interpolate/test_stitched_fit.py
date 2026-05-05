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
