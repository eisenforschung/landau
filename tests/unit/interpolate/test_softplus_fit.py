import numpy as np
from landau.interpolate import SoftplusFit
from hypothesis import given, strategies as st, settings, HealthCheck

@st.composite
def softplus_params(draw):
    n_softplus = draw(st.integers(min_value=1, max_value=3))
    slope = draw(st.floats(min_value=0.1, max_value=10.0))
    offset = draw(st.floats(min_value=-1.0, max_value=1.0))
    return n_softplus, slope, offset

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    params=softplus_params()
)
def test_SoftplusFit_hypothesis(params):
    n_softplus, slope, offset = params
    x = np.linspace(0, 1, 50)
    y = np.log1p(np.exp(slope * (x - 0.5 + offset)))
    sf = SoftplusFit(n_softplus=n_softplus, max_nfev=200)
    fit = sf.fit(x, y)
    assert np.allclose(fit(x), y, atol=0.1)
