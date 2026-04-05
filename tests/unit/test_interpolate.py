import pytest
import numpy as np
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

    T_arr = np.array([0.0, 1.0])
    G_arr = G_calphad(T_arr, 1.0, 2.0, 3.0)
    assert G_arr[0] == 2.0
    # G_calphad(1.0, 1.0, 2.0, 3.0) = 1.0*ln(1.0)*1.0 + 2.0*1.0^0 + 3.0*1.0^1 = 0 + 2 + 3 = 5
    assert G_arr[1] == 5.0

def test_polyfit_basic():
    # Fit y = x^2 with 3 params (a + bx + cx^2)
    x = np.linspace(0, 1, 10)
    y = x**2

    pf = PolyFit(nparam=3)
    fit_func = pf.fit(x, y)

    assert isinstance(fit_func, np.poly1d)
    # Check if fit is accurate
    assert np.isclose(fit_func(0.5), 0.25, atol=1e-5)

def test_polyfit_auto():
    # Fit y = 2x with nparam="auto"
    x = np.linspace(0, 1, 20)
    y = 2 * x

    pf = PolyFit(nparam="auto")
    fit_func = pf.fit(x, y)

    assert isinstance(fit_func, np.poly1d)
    assert np.isclose(fit_func(0.7), 1.4, atol=1e-5)

def test_polyfit_curvature_warning():
    # Fit with enforce_curvature=True when polyfit package is missing
    # If polyfit is installed, this test should be skipped or handle it
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
    pl_true = -0.5
    p0_true = 1.2
    p1_true = 0.8
    y = G_calphad(T, pl_true, p0_true, p1_true)

    sgte = SGTE(nparam=3)
    fit_func = sgte.fit(T, y)

    assert np.isclose(fit_func(5.0), G_calphad(5.0, pl_true, p0_true, p1_true), atol=1e-5)

def test_redlich_kister_basic():
    # Fit y = 0.5 * c * (1-c) + 2.0 * c + 1.0
    # RedlichKisterInterpolation: f(c) = pre * sum(L_i * (2x-1)^i) + f0 + df * c
    # Let L = [0.5], f0 = 1.0, df = 2.0
    # f(c) = c(1-c)*0.5 + 1.0 + 2.0*c
    c = np.linspace(0, 1, 10)
    y = c * (1 - c) * 0.5 + 1.0 + 2.0 * c

    rk = RedlichKister(nparam=1)
    fit_func = rk.fit(c, y)

    assert isinstance(fit_func, RedlichKisterInterpolation)
    assert np.isclose(fit_func(0.3), 0.3 * (1 - 0.3) * 0.5 + 1.0 + 2.0 * 0.3, atol=1e-5)

    # Test terminal checks
    with pytest.raises(AssertionError, match="Must include terminals"):
        rk.fit(c[1:-1], y[1:-1])

def test_stitched_fit_basic():
    # SGTE for middle, PolyFit for boundaries
    T = np.linspace(100, 1000, 50)
    # y = T*ln(T)
    y = T * np.log(T)

    # We want to check if the boundaries are correctly used.
    # StitchedFit.fit(t, f) uses self.low for t < tmin and self.upp for t > tmax.
    # The internal `mid` is fit on all (t, f).

    # Let's use a PolyFit(2) (linear) for 'upp' fit on the last 'edge' points.
    # If we extrapolate beyond T=1000, it should be linear.
    sf = StitchedFit(
        interpolating=SGTE(2), # p0 + p1*T (not a great fit for TlnT, but it works for testing stitching)
        upp=PolyFit(2),        # linear fit for T > 1000
        edge=5
    )

    fit_res = sf.fit(T, y)

    # T=1000 is the max. T=1100 should use the PolyFit(2) extrapolated from last 5 points.
    # T=500 should use SGTE(2) fit on all T.

    # Verify it doesn't crash
    val_mid = fit_res(500)
    val_upp = fit_res(1100)

    assert isinstance(val_mid, (float, np.floating))
    assert isinstance(val_upp, (float, np.floating))

def test_softplus_fit_basic():
    # Fit y = x
    x = np.linspace(0, 1, 20)
    y = x

    spf = SoftplusFit(n_softplus=1)
    fit_func = spf.fit(x, y)

    # Check if fit is reasonable
    assert np.isclose(fit_func(0.5), 0.5, atol=0.05)
