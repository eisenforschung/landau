import numpy as np
from landau.interpolate import G_calphad
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@given(
    pl=st.floats(min_value=-1e2, max_value=1e2),
    p=arrays(dtype=float, shape=st.integers(min_value=1, max_value=5), elements=st.floats(min_value=-1e2, max_value=1e2))
)
def test_G_calphad_hypothesis(pl, p):
    # Test scalar T
    T_scalar = 1.5
    res_scalar = G_calphad(T_scalar, pl, *p)
    expected_scalar = T_scalar * np.log(T_scalar) * pl + sum(pi * T_scalar**i for i, pi in enumerate(p))
    assert np.isclose(res_scalar, expected_scalar)

    # Test array T
    T_array = np.array([1.5, 2.5])
    res_array = G_calphad(T_array, pl, *p)
    expected_array = T_array * np.log(T_array) * pl + np.array([sum(pi * t**i for i, pi in enumerate(p)) for t in T_array])
    assert np.allclose(res_array, expected_array)
    assert np.isclose(res_array[0], res_scalar)

@given(
    pl=st.floats(min_value=-1e2, max_value=1e2),
    p=arrays(dtype=float, shape=st.integers(min_value=1, max_value=5), elements=st.floats(min_value=-1e2, max_value=1e2))
)
def test_G_calphad_at_zero(pl, p):
    # T=0 edge case should return p[0]
    assert np.isclose(G_calphad(0.0, pl, *p), p[0])
    assert np.allclose(G_calphad(np.array([0.0, 1.0]), pl, *p), [p[0], G_calphad(1.0, pl, *p)])
