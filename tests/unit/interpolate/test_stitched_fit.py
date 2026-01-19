import numpy as np
from landau.interpolate import StitchedFit, PolyFit
from hypothesis import given, strategies as st

@given(
    t_min=st.floats(min_value=0.0, max_value=0.4),
    t_max=st.floats(min_value=0.6, max_value=1.0),
    a=st.floats(min_value=-1, max_value=1),
    b=st.floats(min_value=-1, max_value=1),
    c=st.floats(min_value=-1, max_value=1),
    edge=st.integers(min_value=10, max_value=20)
)
def test_StitchedFit_hypothesis(t_min, t_max, a, b, c, edge):
    t = np.linspace(t_min, t_max, 100)
    def func(x):
        return a * x**2 + b * x + c
    y = func(t)
    stitched = StitchedFit(
        interpolating=PolyFit(nparam=3),
        low=PolyFit(nparam=3),
        upp=PolyFit(nparam=3),
        edge=edge
    )
    fit = stitched.fit(t, y)
    assert np.allclose(fit(t), y, atol=1e-3)
    t_upp = np.linspace(t_max, t_max + 0.1, 10)
    assert np.allclose(fit(t_upp), func(t_upp), atol=1e-2)
    t_low = np.linspace(t_min - 0.1, t_min, 10)
    assert np.allclose(fit(t_low), func(t_low), atol=1e-2)
    assert np.isclose(fit(t_min), fit(t_min - 1e-6), atol=1e-3)
    assert np.isclose(fit(t_max), fit(t_max + 1e-6), atol=1e-3)
