"""Tests for the temperature-aware 2-D softplus surface interpolator.

The scenarios mirror the CuAu intermetallic validation notebook
``phase_diagrams_2d_softplus_surface``: functional recovery of a known surface,
the convexity guarantee that motivates a softplus (vs polynomial) amplitude, the
analytic c-derivative, and end-to-end use inside ``Surface2DInterpolatingPhase``.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from landau.interpolate import (
    SoftplusSurface2DInterpolator,
    SoftplusFittedSurface,
)
from landau.interpolate.softplus import _softplus
from landau.phases import Surface2DInterpolatingPhase, TemperatureDependentLinePhase, S
from landau.interpolate import SGTE

# Shared tolerances.
RECOVER_ATOL = 1e-4   # functional recovery of an in-family surface
DERIV_ATOL = 1e-6     # analytic vs finite-difference c-derivative
CONVEX_ATOL = 1e-9    # second-difference convexity floor (numerical slack)
GIBBS_ATOL = 5e-4     # c = -dphi/dmu via the analytic slice derivative


def _softplus_surface(T, c, *, amps, slopes, knees, offset):
    """Evaluate a ground-truth sum-of-softplus surface (entropy-removed H).

    ``amps``/``slopes``/``knees`` are callables of T (broadcast over the flat
    sample arrays); amplitudes must be non-negative for a convex surface.
    """
    H = np.full(np.shape(c), offset, float)
    for a, b, k in zip(amps, slopes, knees):
        H = H + a(T) * _softplus(b(T) * (c + k(T)))
    return H


def _grid(Tlo=400.0, Thi=800.0, nT=30, clo=0.2, chi=0.8, nc=9):
    Tg = np.linspace(Tlo, Thi, nT)
    cg = np.linspace(clo, chi, nc)
    T = np.repeat(Tg, nc)
    c = np.tile(cg, nT)
    return T, c, Tg, cg


def _second_difference_min(f, lo, hi, n=120):
    c = np.linspace(lo, hi, n)
    y = np.asarray(f(c), float)
    return float((y[2:] - 2 * y[1:-1] + y[:-2]).min())


# --------------------------------------------------------------------------- #
# functional recovery
# --------------------------------------------------------------------------- #
def test_recovers_in_family_surface():
    """A surface that *is* a T-modulated sum of softpluses is recovered tightly."""
    T, c, Tg, cg = _grid()
    amps = [lambda t: 0.05 + 2e-5 * (t - 600), lambda t: 0.03]
    slopes = [lambda t: 4.0 + 1e-3 * (t - 600), lambda t: -3.0 - 5e-4 * (t - 600)]
    knees = [lambda t: -0.5, lambda t: -0.5]
    H = _softplus_surface(T, c, amps=amps, slopes=slopes, knees=knees, offset=-0.2)

    surface = SoftplusSurface2DInterpolator(n_softplus=2).fit(T, c, H)
    assert isinstance(surface, SoftplusFittedSurface)

    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        Hq = _softplus_surface(Tq, cg, amps=amps, slopes=slopes, knees=knees, offset=-0.2)
        np.testing.assert_allclose(surface.slice_at(Tq)(cg), Hq, atol=RECOVER_ATOL)


def test_recovers_better_than_constant_predictor():
    """Even on a non-in-family convex surface the fit beats the mean predictor."""
    T, c, Tg, cg = _grid()
    H = 0.4 * (c - 0.55) ** 2 + 2e-5 * (T - 600) * (c - 0.55) ** 2
    surface = SoftplusSurface2DInterpolator().fit(T, c, H)
    pred = np.concatenate([surface.slice_at(Tq)(cg) for Tq in Tg])
    mse_fit = np.mean((pred - H) ** 2)
    mse_mean = np.mean((H - H.mean()) ** 2)
    assert mse_fit < 1e-3 * mse_mean


# --------------------------------------------------------------------------- #
# convexity (the reason softplus is used over a polynomial amplitude)
# --------------------------------------------------------------------------- #
@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    Tq=st.floats(min_value=300.0, max_value=1500.0),  # incl. extrapolation past 800 K
)
def test_fitted_slice_is_convex_even_on_noisy_data(seed, Tq):
    """Amplitudes are non-negative and every slice is convex, by construction --
    holds for arbitrary data, so a cheap under-converged fit still suffices."""
    rng = np.random.default_rng(seed)
    T, c, Tg, cg = _grid(nT=15, nc=7)
    H = rng.normal(scale=0.05, size=c.shape)  # arbitrary, non-convex data
    surface = SoftplusSurface2DInterpolator(max_nfev=300).fit(T, c, H)
    sl = surface.slice_at(Tq)
    assert (sl.a >= 0).all()
    assert _second_difference_min(sl, 0.2, 0.8) >= -CONVEX_ATOL


def test_convexity_holds_far_outside_training_range():
    """The softplus amplitude link keeps slices convex under heavy extrapolation."""
    T, c, Tg, cg = _grid()
    H = 0.4 * (c - 0.5) ** 2 + 1e-4 * (T - 600) * (c - 0.5)
    surface = SoftplusSurface2DInterpolator().fit(T, c, H)
    for Tq in (50.0, 5000.0):
        sl = surface.slice_at(Tq)
        assert (sl.a >= 0).all()
        assert _second_difference_min(sl, 0.2, 0.8) >= -CONVEX_ATOL


# --------------------------------------------------------------------------- #
# analytic c-derivative
# --------------------------------------------------------------------------- #
def test_slice_analytic_derivative_matches_finite_difference():
    T, c, Tg, cg = _grid()
    H = 0.4 * (c - 0.5) ** 2 + 2e-5 * (T - 600) * (c - 0.5) ** 2
    surface = SoftplusSurface2DInterpolator().fit(T, c, H)
    cc = np.linspace(0.25, 0.75, 50)
    h = 1e-6
    for Tq in (450.0, 650.0, 790.0):
        sl = surface.slice_at(Tq)
        fd = (sl(cc + h) - sl(cc - h)) / (2 * h)
        np.testing.assert_allclose(sl.deriv()(cc), fd, atol=DERIV_ATOL)


# --------------------------------------------------------------------------- #
# shape contract
# --------------------------------------------------------------------------- #
def test_slice_scalar_and_array_shapes():
    T, c, Tg, cg = _grid()
    H = 0.4 * (c - 0.5) ** 2
    sl = SoftplusSurface2DInterpolator().fit(T, c, H).slice_at(600.0)

    scalar = sl(0.5)
    assert np.isscalar(scalar) or (isinstance(scalar, np.ndarray) and scalar.ndim == 0)
    assert np.ndim(scalar) == 0

    arr = sl(np.linspace(0.3, 0.7, 11))
    assert isinstance(arr, np.ndarray) and arr.shape == (11,)

    deriv_scalar = sl.deriv()(0.5)
    assert np.ndim(deriv_scalar) == 0


# --------------------------------------------------------------------------- #
# temperature dependence is real
# --------------------------------------------------------------------------- #
def test_slice_tracks_temperature():
    """A surface whose curvature grows with T yields T-dependent slices that each
    match their own data slice."""
    T, c, Tg, cg = _grid()
    # curvature scales with T -> the fixed-T slices genuinely differ
    H = (0.2 + 6e-4 * (T - 400)) * (c - 0.5) ** 2
    surface = SoftplusSurface2DInterpolator().fit(T, c, H)
    lo = surface.slice_at(400.0)(cg)
    hi = surface.slice_at(800.0)(cg)
    # the two slices are not the same curve
    assert np.abs(hi - lo).max() > 1e-2
    # each reproduces its own data slice
    np.testing.assert_allclose(lo, 0.2 * (cg - 0.5) ** 2, atol=2e-3)
    np.testing.assert_allclose(hi, (0.2 + 6e-4 * 400) * (cg - 0.5) ** 2, atol=2e-3)


# --------------------------------------------------------------------------- #
# input validation, immutability
# --------------------------------------------------------------------------- #
def test_fit_requires_two_distinct_concentrations():
    T = np.linspace(400, 800, 20)
    c = np.full_like(T, 0.5)
    f = np.zeros_like(T)
    with pytest.raises(ValueError, match="two distinct concentrations"):
        SoftplusSurface2DInterpolator().fit(T, c, f)


def test_interpolator_is_frozen_and_hashable():
    a = SoftplusSurface2DInterpolator()
    b = SoftplusSurface2DInterpolator()
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert SoftplusSurface2DInterpolator(b_order=3) != a
    with pytest.raises(Exception):
        a.n_softplus = 5


# --------------------------------------------------------------------------- #
# integration with Surface2DInterpolatingPhase
# --------------------------------------------------------------------------- #
def _intermetallic_line_phases(clo=0.30, chi=0.40, ncomp=7, depth=1.0):
    """Narrow-window line phases with a convex, mildly T-dependent excess.

    Mimics an intermetallic: a few compositions over a small window, each with a
    reversible-scaling temperature sweep (entropy already included, so the phase
    is built with ``add_entropy=False``).  ``depth`` is chosen so the well is
    curved enough that the equilibrium ``c`` sweeps the *interior* of the window
    as ``dmu`` varies (rather than pinning at an edge).
    """
    Tsweep = np.linspace(400.0, 800.0, 60)
    cs = np.linspace(clo, chi, ncomp)
    c0 = 0.5 * (clo + chi)
    lines = []
    for ci in cs:
        # entropy-removed excess H_i(T) = depth*(1 + 1e-4 T)*(c-c0)^2
        H = depth * (1 + 1e-4 * Tsweep) * (ci - c0) ** 2
        f = H - Tsweep * S(np.array(ci))   # store free energy WITH entropy
        lines.append(TemperatureDependentLinePhase(
            "im", float(ci), Tsweep, f, interpolator=SGTE(3)))
    return lines, (float(clo), float(chi))


def test_surface_phase_solver_and_gibbs_duhem():
    lines, crange = _intermetallic_line_phases()
    phase = Surface2DInterpolatingPhase(
        "im", lines, concentration_range=crange, add_entropy=False,
        surface_interpolator=SoftplusSurface2DInterpolator())

    T = 500.0
    dmu = np.linspace(-0.2, 0.2, 41)
    c = phase.concentration(np.full_like(dmu, T), dmu)
    phi = phase.semigrand_potential(np.full_like(dmu, T), dmu)
    assert np.isfinite(c).all() and np.isfinite(phi).all()
    assert (c >= crange[0] - 1e-6).all() and (c <= crange[1] + 1e-6).all()

    # c = -dphi/ddmu via the analytic slice derivative, at strictly interior c
    h = 1e-4
    cphi = -(phase.semigrand_potential(np.full_like(dmu, T), dmu + h)
             - phase.semigrand_potential(np.full_like(dmu, T), dmu - h)) / (2 * h)
    interior = (c > crange[0] + 1e-3) & (c < crange[1] - 1e-3)
    assert interior.any()
    np.testing.assert_allclose(c[interior], cphi[interior], atol=GIBBS_ATOL)


def test_surface_phase_free_energy_is_convex():
    """No spurious miscibility gap: f(c) is convex across the window and T."""
    lines, crange = _intermetallic_line_phases()
    phase = Surface2DInterpolatingPhase(
        "im", lines, concentration_range=crange, add_entropy=False,
        surface_interpolator=SoftplusSurface2DInterpolator())
    cc = np.linspace(crange[0] + 1e-3, crange[1] - 1e-3, 150)
    for T in (400.0, 600.0, 800.0):
        f = phase.free_energy(T, cc)
        d2 = f[2:] - 2 * f[1:-1] + f[:-2]
        assert d2.min() >= -CONVEX_ATOL
