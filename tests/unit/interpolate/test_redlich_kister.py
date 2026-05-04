import numpy as np
import pytest
from landau.interpolate import RedlichKister, RedlichKisterInterpolation
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

def test_RedlichKister_terminal_check():
    rk = RedlichKister(nparam=2)
    c_no_term = np.linspace(0.1, 0.9, 10)
    y_no_term = np.zeros_like(c_no_term)
    with pytest.raises(AssertionError, match="Must include terminals when fitting Redlich-Kister!"):
        rk.fit(c_no_term, y_no_term)

@given(
    L=arrays(dtype=float, shape=st.integers(min_value=1, max_value=3), elements=st.floats(min_value=-1, max_value=1)),
    f0=st.floats(min_value=-1, max_value=1),
    df=st.floats(min_value=-1, max_value=1)
)
def test_RedlichKister_hypothesis(L, f0, df):
    c = np.linspace(0, 1, 20)
    y_mix = RedlichKisterInterpolation._eval_mix(c, *L)
    y = y_mix + f0 + df * c
    rk = RedlichKister(nparam=len(L))
    fit = rk.fit(c, y)
    assert np.allclose(fit(c), y, atol=1e-5)
    assert np.isclose(fit.f0, f0, atol=1e-5)
    assert np.isclose(fit.df, df, atol=1e-5)
    assert np.allclose(fit.rk_parameters, L, atol=1e-5)
