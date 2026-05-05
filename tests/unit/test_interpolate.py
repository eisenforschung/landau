import pytest
import numpy as np
from unittest.mock import MagicMock
from landau.interpolate import (
    G_calphad,
    PolyFit,
    SGTE,
    RedlichKister,
    StitchedFit,
    SoftplusFit,
    RedlichKisterInterpolation,
)

def test_g_calphad():
    # Test T=0 cases (should return p[0])
    assert G_calphad(0.0, 1.0, 2.0, 3.0) == 2.0

    T_arr = np.array([0.0, 1.0, 2.0])
    G_arr = G_calphad(T_arr, 1.0, 2.0, 3.0)
    assert G_arr[0] == 2.0
    # G_calphad(1.0, 1.0, 2.0, 3.0) = 1.0*ln(1.0)*1.0 + 2.0 + 3.0 = 5.0
    assert np.isclose(G_arr[1], 5.0)
    # G_calphad(2.0, 1.0, 2.0, 3.0) = 2.0*ln(2.0)*1.0 + 2.0 + 3.0*2.0 = 2*ln(2) + 8
    assert np.isclose(G_arr[2], 2*np.log(2) + 8)

def test_polyfit_basic():
    # Fit y = 1 + 2x + 3x^2
    x = np.linspace(0, 1, 10)
    y = 1 + 2*x + 3*x**2

    pf = PolyFit(nparam=3)
    fit_func = pf.fit(x, y)

    assert isinstance(fit_func, np.poly1d)
    # Verify coefficients (stored in descending order in np.poly1d: c, b, a)
    assert np.allclose(fit_func.coeffs, [3, 2, 1], atol=1e-5)
    # Check multiple points
    test_x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(fit_func(test_x), 1 + 2*test_x + 3*test_x**2)

def test_polyfit_auto():
    # Fit a constant y = 5. L1 should ideally pick only the first param.
    x = np.linspace(0, 1, 20)
    y = np.full_like(x, 5.0)

    pf = PolyFit(nparam="auto")
    fit_func = pf.fit(x, y)

    assert isinstance(fit_func, np.poly1d)
    assert np.isclose(fit_func(0.5), 5.0, atol=1e-5)
    # Should have few non-zero coefficients
    assert len(np.where(abs(fit_func.coeffs) > 1e-5)[0]) <= 2

def test_polyfit_curvature_warning():
    # Fit with enforce_curvature=True when polyfit package is missing
    from landau.interpolate import polyfit
    if polyfit is not None:
        pytest.skip("polyfit is installed, skipping warning test")

    x = np.linspace(0, 1, 10)
    y = -x**2 # concave

    pf = PolyFit(nparam=3, enforce_curvature=True)
    with pytest.warns(UserWarning, match="enforce_curvature=True is only supported"):
        pf.fit(x, y)

def test_sgte_basic():
    # Fit y = G_calphad(T, ...)
    T = np.linspace(0.1, 10, 20)
    pl_true, p0_true, p1_true = -0.5, 1.2, 0.8
    y = G_calphad(T, pl_true, p0_true, p1_true)

    sgte = SGTE(nparam=3)
    fit_func = sgte.fit(T, y)

    # Check recovery of values over a range
    test_T = np.array([1.0, 5.0, 9.0])
    assert np.allclose(fit_func(test_T), G_calphad(test_T, pl_true, p0_true, p1_true), atol=1e-5)

def test_redlich_kister_basic():
    # f(c) = pre * sum(Li * (2x-1)^i) + f0 + df * c
    # Let L = [10.0], f0 = 1.0, df = 2.0
    # f(c) = c(1-c)*10.0 + 1.0 + 2.0*c
    c = np.linspace(0, 1, 11)
    y = c * (1 - c) * 10.0 + 1.0 + 2.0 * c

    rk = RedlichKister(nparam=1)
    fit_func = rk.fit(c, y)

    assert isinstance(fit_func, RedlichKisterInterpolation)
    assert np.isclose(fit_func.f0, 1.0, atol=1e-5)
    assert np.isclose(fit_func.df, 2.0, atol=1e-5)
    assert np.allclose(fit_func.rk_parameters, [10.0], atol=1e-5)

    # Test call
    assert np.allclose(fit_func(c), y, atol=1e-5)

    # Test terminal checks
    with pytest.raises(AssertionError, match="Must include terminals"):
        rk.fit(c[1:-1], y[1:-1])

def test_stitched_fit_advanced():
    # Define three very distinct regions
    # low: y = -100
    # mid: y = 0
    # upp: y = 100
    T = np.linspace(10, 20, 50)
    y = np.zeros_like(T)

    # We use PolyFit(1) for low/upp which will return constant functions if fitted on constant data
    mock_low = PolyFit(nparam=1)
    mock_upp = PolyFit(nparam=1)
    mock_mid = PolyFit(nparam=1)

    sf = StitchedFit(
        interpolating=mock_mid,
        low=mock_low,
        upp=mock_upp,
        edge=5
    )

    # Construct data such that edge regions are distinct
    y[:5] = -100
    y[5:-5] = 0
    y[-5:] = 100

    fit_res = sf.fit(T, y)

    # Test regions
    assert np.isclose(fit_res(5), -100)   # T < Tmin=10
    assert np.isclose(fit_res(15), 0)    # Tmin < T < Tmax
    assert np.isclose(fit_res(25), 100)   # T > Tmax=20

    # Test array input
    test_T = np.array([5, 15, 25])
    assert np.allclose(fit_res(test_T), [-100, 0, 100])

def test_softplus_fit_robustness():
    # Fit a more complex shape: a step-like function
    x = np.linspace(-5, 5, 100)
    y = np.where(x < 0, 0, 1) # Heaviside-like

    # Softplus can approximate this with enough terms
    spf = SoftplusFit(n_softplus=2, max_nfev=500)
    fit_func = spf.fit(x, y)

    # Check if it captures the trend
    assert fit_func(-4) < 0.2
    assert fit_func(4) > 0.8
    assert fit_func(-10) < fit_func(10) # monotonically increasing trend captured

def test_redlich_kister_static_methods():
    # _eval_mix: pre * sum(Li * (2x-1)^i)
    L = np.array([1.0, 2.0])
    x = 0.7
    # pre = 0.7 * 0.3 = 0.21
    # xi = 2*0.7 - 1 = 0.4
    # sum = 1.0 * 0.4^0 + 2.0 * 0.4^1 = 1.0 + 0.8 = 1.8
    # result = 0.21 * 1.8 = 0.378
    val = RedlichKisterInterpolation._eval_mix(x, *L)
    assert np.isclose(val, 0.378)

    # _eval_mix_derivative
    # f(x) = x(1-x) * (L0 + L1(2x-1)) = (x-x^2) * (L0 + 2*L1*x - L1)
    # f'(x) = (1-2x)*(L0 + 2*L1*x - L1) + (x-x^2)*(2*L1)
    # For x=0.7, L=[1.0, 2.0]:
    # f'(0.7) = (1-1.4)*(1.0 + 2*2*0.7 - 2.0) + (0.7-0.49)*(2*2)
    # f'(0.7) = (-0.4)*(1.0 + 2.8 - 2.0) + (0.21)*(4)
    # f'(0.7) = (-0.4)*(1.8) + 0.84 = -0.72 + 0.84 = 0.12
    deriv = RedlichKisterInterpolation._eval_mix_derivative(x, *L)
    assert np.isclose(deriv, 0.12)
