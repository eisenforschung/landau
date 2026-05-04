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
