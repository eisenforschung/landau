import numpy as np
import pytest
from landau.interpolate import SplineFit
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

# Shared tolerance for the strict-interpolation assertions.  A degree-k spline
# reproduces degree-k polynomials to roundoff (~1e-14 over random ±10 cubics),
# so 1e-9 keeps margin while staying far below anything a constant/linear fit
# could reach.
ATOL = 1e-9


def test_SplineFit_interpolates_exactly():
    """smoothing=0 makes the spline pass through every sample."""
    x = np.linspace(0, 1, 12)
    y = np.cos(3 * x) + 0.5 * x
    fit = SplineFit(degree=3, smoothing=0.0).fit(x, y)
    assert np.max(np.abs(fit(x) - y)) < ATOL


@given(
    coeffs=arrays(dtype=float, shape=4, elements=st.floats(min_value=-10, max_value=10)),
)
def test_SplineFit_reproduces_cubic(coeffs):
    """A cubic interpolating spline recovers any cubic, including between nodes."""
    x = np.linspace(0, 1, 8)
    y = np.polyval(coeffs, x)
    fit = SplineFit(degree=3, smoothing=0.0).fit(x, y)
    xt = np.linspace(0, 1, 101)
    assert np.allclose(fit(xt), np.polyval(coeffs, xt), atol=ATOL)


def test_SplineFit_sorts_unordered_input():
    """The fit sorts x internally, so shuffled samples interpolate identically."""
    x = np.linspace(0, 1, 9)
    y = np.exp(x) - x**2
    order = np.array([4, 0, 8, 2, 6, 1, 7, 3, 5])
    fit = SplineFit(degree=3, smoothing=0.0).fit(x[order], y[order])
    assert np.max(np.abs(fit(x) - y)) < ATOL


def test_SplineFit_scalar_and_array_shapes():
    """Scalar in -> python float out; array in -> array out of matching shape."""
    x = np.linspace(0, 1, 10)
    y = x**2
    fit = SplineFit().fit(x, y)
    out = fit(0.5)
    assert isinstance(out, float)
    assert np.isclose(out, 0.25, atol=ATOL)
    c = np.array([0.1, 0.4, 0.9])
    assert fit(c).shape == c.shape


def test_SplineFit_degree_one_is_piecewise_linear():
    """degree=1 is plain piecewise-linear interpolation, matching np.interp."""
    x = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    y = np.array([1.0, -0.3, 0.8, 0.1, 2.0])
    fit = SplineFit(degree=1, smoothing=0.0).fit(x, y)
    xt = np.linspace(0, 1, 101)
    assert np.allclose(fit(xt), np.interp(xt, x, y), atol=ATOL)


def test_SplineFit_degree_clamped_to_data():
    """Fewer samples than degree+1 drop k to len(x)-1 (here two points -> linear)."""
    x = np.array([0.2, 0.8])
    y = np.array([1.0, 2.0])
    fit = SplineFit(degree=3, smoothing=0.0).fit(x, y)
    # linear through the two points: midpoint is the average
    assert np.isclose(fit(0.5), 1.5, atol=ATOL)
    assert np.isclose(fit(0.2), 1.0, atol=ATOL)
    assert np.isclose(fit(0.8), 2.0, atol=ATOL)


def test_SplineFit_smoothing_denoises():
    """smoothing>0 trades node fidelity for a curve closer to the clean signal."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 40)
    clean = np.sin(2 * np.pi * x)
    y = clean + 0.1 * rng.standard_normal(x.size)

    strict = SplineFit(degree=3, smoothing=0.0).fit(x, y)
    s = 0.2
    smooth = SplineFit(degree=3, smoothing=s).fit(x, y)

    # strict interpolation still reproduces the noisy samples exactly ...
    assert np.max(np.abs(strict(x) - y)) < ATOL
    # ... whereas smoothing departs from them ...
    assert np.max(np.abs(smooth(x) - y)) > 1e-2
    # ... up to (and no further than) the requested squared-residual budget ...
    assert np.isclose(np.sum((smooth(x) - y) ** 2), s, rtol=1e-2)
    # ... landing closer to the underlying signal than the strict fit does.
    def rms(fit):
        return np.sqrt(np.mean((fit(x) - clean) ** 2))

    assert rms(smooth) < rms(strict)


def test_SplineFit_none_smoothing_uses_scipy_default():
    """smoothing=None defers to scipy's default factor, which smooths the data."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 40)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
    fit = SplineFit(degree=3, smoothing=None).fit(x, y)
    # default s is the sample count, so the curve is allowed to miss the nodes
    assert np.max(np.abs(fit(x) - y)) > 1e-2


def test_SplineFit_is_hashable_and_immutable():
    """Frozen dataclass: usable as a cache key, fields cannot be reassigned."""
    a = SplineFit(degree=3, smoothing=0.0)
    b = SplineFit(degree=3, smoothing=0.0)
    assert a == b and hash(a) == hash(b)
    assert {a, b} == {a}
    with pytest.raises(Exception):
        a.degree = 2
