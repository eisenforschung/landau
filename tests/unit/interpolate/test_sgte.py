import numpy as np
import pytest
from landau.interpolate import SGTE, SGTEInterpolation, G_calphad
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


def test_deriv_pure_polynomial():
    # G(T) = 5 + 3T + 2T^2  =>  dG/dT = 3 + 4T (pl=0 drops the T*ln(T) term)
    interp = SGTEInterpolation((0.0, 5.0, 3.0, 2.0))
    T = np.array([1.0, 10.0, 100.0])
    assert np.allclose(interp.deriv()(T), 3.0 + 4.0 * T)


def test_deriv_log_branch():
    # G(T) = T*ln(T) (pl=1, no polynomial terms) => dG/dT = ln(T) + 1
    interp = SGTEInterpolation((1.0, 0.0))
    T = np.array([1.0, 10.0, 100.0])
    assert np.allclose(interp.deriv()(T), np.log(T) + 1.0)


def test_deriv_scalar_and_array_shape_contract():
    interp = SGTEInterpolation((0.5, 5.0, 3.0, 2.0))
    scalar_out = interp.deriv()(10.0)
    assert np.ndim(scalar_out) == 0
    assert np.isclose(scalar_out, 0.5 * (np.log(10.0) + 1.0) + 3.0 + 4.0 * 10.0)

    T = np.linspace(50, 500, 5)
    array_out = interp.deriv()(T)
    assert array_out.shape == T.shape
