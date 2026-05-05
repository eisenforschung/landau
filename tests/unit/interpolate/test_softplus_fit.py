import numpy as np
from landau.interpolate import SoftplusFit
from hypothesis import given, strategies as st

@given(
    n_softplus=st.integers(min_value=1, max_value=2),
    slope=st.floats(min_value=0.1, max_value=5.0),
    offset=st.floats(min_value=-0.5, max_value=0.5)
)
def test_SoftplusFit_hypothesis(n_softplus, slope, offset):
    # Use fewer points and smaller max_nfev to keep it fast and avoid Hypothesis deadlines
    x = np.linspace(0, 1, 30)
    y = np.log1p(np.exp(slope * (x - 0.5 + offset)))
    sf = SoftplusFit(n_softplus=n_softplus, max_nfev=50)
    fit = sf.fit(x, y)
    assert np.allclose(fit(x), y, atol=0.2)


def test_SoftplusFit_step_function():
    """SoftplusFit should fit a soft step (Heaviside-like) better than a constant predictor."""
    x = np.linspace(0, 1, 60)
    y = 1.0 / (1.0 + np.exp(-20 * (x - 0.5)))  # sigmoid step at 0.5
    sf = SoftplusFit(n_softplus=2, max_nfev=500)
    fit = sf.fit(x, y)
    pred = fit(x)

    # Must do better than predicting the mean
    mse_fit = np.mean((pred - y) ** 2)
    mse_mean = np.mean((np.full_like(y, y.mean()) - y) ** 2)
    assert mse_fit < mse_mean

    # Should capture the low and high plateaus
    assert pred[0] < 0.3
    assert pred[-1] > 0.7


def test_SoftplusFit_monotone_data():
    """SoftplusFit should recover a simple monotone ramp reasonably well."""
    x = np.linspace(0, 1, 40)
    y = x  # linear ramp
    sf = SoftplusFit(n_softplus=1, max_nfev=200)
    fit = sf.fit(x, y)
    np.testing.assert_allclose(fit(x), y, atol=0.1)
