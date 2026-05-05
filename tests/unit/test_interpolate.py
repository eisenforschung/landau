import importlib.util

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
    assert G_calphad(0.0, 1.0, 2.0, 3.0) == 2.0

    T_arr = np.array([0.0, 1.0, 2.0])
    G_arr = G_calphad(T_arr, 1.0, 2.0, 3.0)
    assert G_arr[0] == 2.0
    # G_calphad(1.0, 1.0, 2.0, 3.0) = 1.0*ln(1.0)*1.0 + 2.0 + 3.0 = 5.0
    assert np.isclose(G_arr[1], 5.0)
    # G_calphad(2.0, 1.0, 2.0, 3.0) = 2.0*ln(2.0)*1.0 + 2.0 + 3.0*2.0 = 2*ln(2) + 8
    assert np.isclose(G_arr[2], 2 * np.log(2) + 8)


def test_polyfit_basic():
    x = np.linspace(0, 1, 10)
    y = 1 + 2 * x + 3 * x**2

    pf = PolyFit(nparam=3)
    fit_func = pf.fit(x, y)

    assert isinstance(fit_func, np.poly1d)
    # Coefficients stored in descending order: [c, b, a] for a + bx + cx^2
    assert np.allclose(fit_func.coeffs, [3, 2, 1], atol=1e-5)
    test_x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(fit_func(test_x), 1 + 2 * test_x + 3 * test_x**2)


def test_polyfit_auto():
    x = np.linspace(0, 1, 20)
    y = np.full_like(x, 5.0)

    pf = PolyFit(nparam="auto")
    fit_func = pf.fit(x, y)

    assert isinstance(fit_func, np.poly1d)
    assert np.isclose(fit_func(0.5), 5.0, atol=1e-5)
    assert len(np.where(abs(fit_func.coeffs) > 1e-5)[0]) <= 2


def test_polyfit_curvature_warning():
    if importlib.util.find_spec("polyfit") is not None:
        pytest.skip("polyfit is installed, skipping warning test")

    x = np.linspace(0, 1, 10)
    y = -x**2

    pf = PolyFit(nparam=3, enforce_curvature=True)
    with pytest.warns(UserWarning, match="enforce_curvature=True is only supported"):
        pf.fit(x, y)


def test_sgte_basic():
    T = np.linspace(0.1, 10, 20)
    pl_true, p0_true, p1_true = -0.5, 1.2, 0.8
    y = G_calphad(T, pl_true, p0_true, p1_true)

    sgte = SGTE(nparam=3)
    fit_func = sgte.fit(T, y)

    test_T = np.array([1.0, 5.0, 9.0])
    assert np.allclose(fit_func(test_T), G_calphad(test_T, pl_true, p0_true, p1_true), atol=1e-5)


def test_redlich_kister_basic():
    c = np.linspace(0, 1, 11)
    y = c * (1 - c) * 10.0 + 1.0 + 2.0 * c

    rk = RedlichKister(nparam=1)
    fit_func = rk.fit(c, y)

    assert isinstance(fit_func, RedlichKisterInterpolation)
    assert np.isclose(fit_func.f0, 1.0, atol=1e-5)
    assert np.isclose(fit_func.df, 2.0, atol=1e-5)
    assert np.allclose(fit_func.rk_parameters, [10.0], atol=1e-5)
    assert np.allclose(fit_func(c), y, atol=1e-5)

    with pytest.raises(AssertionError, match="Must include terminals"):
        rk.fit(c[1:-1], y[1:-1])


def test_redlich_kister_static_methods():
    L = np.array([1.0, 2.0])
    x = 0.7
    # pre = 0.7 * 0.3 = 0.21, xi = 2*0.7 - 1 = 0.4
    # sum = 1.0 * 0.4^0 + 2.0 * 0.4^1 = 1.8 → result = 0.21 * 1.8 = 0.378
    assert np.isclose(RedlichKisterInterpolation._eval_mix(x, *L), 0.378)

    # f'(0.7) = (-0.4)*(1.0 + 2.8 - 2.0) + (0.21)*(4) = -0.72 + 0.84 = 0.12
    assert np.isclose(RedlichKisterInterpolation._eval_mix_derivative(x, *L), 0.12)


def test_stitched_fit_dispatch():
    """Verify StitchedFit routes inputs to low/mid/upp regions."""
    T = np.linspace(10, 20, 50)
    y = np.zeros_like(T)
    y[:5] = -100
    y[5:-5] = 0
    y[-5:] = 100

    # PolyFit(1) is a constant fit; each region returns the mean of its training data
    sf = StitchedFit(
        interpolating=PolyFit(nparam=1),
        low=PolyFit(nparam=1),
        upp=PolyFit(nparam=1),
        edge=5,
    )
    fit_res = sf.fit(T, y)

    assert np.isclose(fit_res(5), -100)   # T < tmin: low= fit
    assert np.isclose(fit_res(15), 0)     # tmin < T < tmax: mid fit
    assert np.isclose(fit_res(25), 100)   # T > tmax: upp= fit

    test_T = np.array([5, 15, 25])
    assert np.allclose(fit_res(test_T), [-100, 0, 100])


def test_stitched_fit_accuracy():
    """Verify StitchedFit mid region accuracy and low= branch."""
    T = np.linspace(100, 1000, 50)
    y = T * np.log(T)

    # SGTE(2) recovers T*ln(T) exactly (pl=1, p0=0)
    sf = StitchedFit(interpolating=SGTE(2), upp=PolyFit(2), edge=5)
    fit_res = sf.fit(T, y)

    assert np.isclose(fit_res(500), 500 * np.log(500), rtol=1e-3)
    val_upp = fit_res(1100)
    assert isinstance(val_upp, (float, np.floating))
    assert not np.isnan(val_upp)

    # low= branch: PolyFit(2) fit on first edge points, extrapolated below tmin
    sf_low = StitchedFit(interpolating=SGTE(2), low=PolyFit(2), upp=None, edge=5)
    fit_low = sf_low.fit(T, y)

    assert np.isclose(fit_low(500), 500 * np.log(500), rtol=1e-3)
    val_low = fit_low(50)
    assert isinstance(val_low, (float, np.floating))
    assert not np.isnan(val_low)


def test_softplus_fit_basic():
    x = np.linspace(0, 1, 50)
    y = x

    # n_softplus=2 has enough parameters to fit a linear function well
    spf = SoftplusFit(n_softplus=2, max_nfev=500)
    fit_func = spf.fit(x, y)

    # Check multiple points — a constant prediction (mean=0.5) fails the near-edge checks
    assert np.isclose(fit_func(0.1), 0.1, atol=0.05)
    assert np.isclose(fit_func(0.5), 0.5, atol=0.05)
    assert np.isclose(fit_func(0.9), 0.9, atol=0.05)

    # Fit must substantially outperform a constant mean prediction
    predictions = np.vectorize(fit_func)(x)
    fit_mse = np.mean((predictions - y) ** 2)
    mean_mse = np.mean((np.mean(y) - y) ** 2)
    assert fit_mse < 0.1 * mean_mse


def test_softplus_fit_robustness():
    x = np.linspace(-5, 5, 100)
    y = np.where(x < 0, 0, 1).astype(float)

    spf = SoftplusFit(n_softplus=2, max_nfev=500)
    fit_func = spf.fit(x, y)

    assert fit_func(-4) < 0.2
    assert fit_func(4) > 0.8
    assert fit_func(-10) < fit_func(10)
