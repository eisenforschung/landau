import numpy as np
import pytest
from landau.interpolate import SGTE, G_calphad
from hypothesis import given, strategies as st

@given(
    pl=st.floats(min_value=1e-5, max_value=1e-3),
    p0=st.floats(min_value=-1, max_value=1),
    p1=st.floats(min_value=-1e-2, max_value=1e-2)
)
def test_SGTE_hypothesis(pl, p0, p1):
    T = np.linspace(100, 1000, 20)
    y = G_calphad(T, pl, p0, p1)
    sgte = SGTE(nparam=3)
    fit = sgte.fit(T, y)
    assert np.allclose(fit(T), y, rtol=1e-2)

def test_SGTE_nparam_check():
    with pytest.raises(AssertionError, match="Must fit at least two parameters!"):
        SGTE(nparam=1)
