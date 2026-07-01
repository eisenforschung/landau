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
from landau.interpolate.softplus import _softplus, _standardize
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


def _v_slope_1overT(T, c, c0=0.36):
    """In-family softplus V whose sharpness b(T) is a degree-1 polynomial in 1/T."""
    b = 4.0 + 30.0 * (1000.0 / np.asarray(T, float))
    u = np.asarray(c, float) - c0
    return -0.2 + 0.05 * _softplus(b * u) + 0.03 * _softplus(-b * u)


def test_recovers_slope_polynomial_in_inverse_T():
    """The slope ``b`` is a polynomial in 1/T, so a well that sharpens as 1/T over
    a wide T-range is recovered to machine precision -- the regime a polynomial in
    T cannot represent.  A poly-in-T slope leaves ~9% error at the extremes here."""
    Tg = np.linspace(200.0, 1000.0, 40)
    cg = np.linspace(0.30, 0.45, 11)
    T = np.repeat(Tg, len(cg))
    c = np.tile(cg, len(Tg))
    surface = SoftplusSurface2DInterpolator(n_softplus=2).fit(T, c, _v_slope_1overT(T, c))
    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        np.testing.assert_allclose(surface.slice_at(Tq)(cg), _v_slope_1overT(Tq, cg), atol=1e-7)


def test_handles_strong_temperature_sharpening():
    """A sharp, asymmetric V whose branch slopes scale with 1/T over a wide T-range
    is fit accurately at *every* temperature -- including the cold extreme where a
    central-slice-only seed or a poly-in-T slope places the knee and branch slopes
    badly.  Pins the fix for the bad-minima behaviour: a degenerate/stuck fit would
    blow the normalised RMSE or the knee position well past these bounds."""
    c0 = 0.36
    Tg = np.linspace(150.0, 1500.0, 60)
    cg = np.linspace(0.30, 0.45, 13)
    T = np.repeat(Tg, len(cg))
    c = np.tile(cg, len(Tg))

    def H(Tq, cq):
        Tq, u = np.asarray(Tq, float), np.asarray(cq, float) - c0
        sL, sR = 0.03 + 0.22 * (1000.0 / Tq), 0.01 + 0.09 * (1000.0 / Tq)
        return np.where(u < 0, sL * (-u), sR * u)

    surface = SoftplusSurface2DInterpolator(n_softplus=2).fit(T, c, H(T, c))
    cd = np.linspace(cg.min(), cg.max(), 401)
    for Tq in Tg:
        depth = max(H(Tq, cg.max()), H(Tq, cg.min()))
        pred = np.asarray(surface.slice_at(Tq)(cg))
        nrmse = np.sqrt(np.mean((pred - H(Tq, cg)) ** 2)) / depth
        knee_err = abs(cd[np.argmin(surface.slice_at(Tq)(cd))] - c0)
        assert nrmse < 1e-3
        assert knee_err < 5e-3


def test_seed_knee_debiased_against_linear_tilt():
    """The per-slice seed places the knee at the curvature centre, found from the
    data with its best-fit line in c removed.  The entropy-removed free energy is a
    small convex dimple on a chemical-potential ramp, where a raw argmin returns
    the downhill window edge; the detrended estimate recovers the knee."""
    from landau.interpolate.softplus import _knee_position

    cn = np.linspace(-1.7, 1.7, 13)
    H = 0.02 * (_softplus(8 * (cn - 0.3)) + _softplus(-8 * (cn - 0.3))) - 0.5 * cn
    assert abs(cn[np.argmin(H)] - 0.3) > 1.0          # raw argmin is fooled by the tilt
    assert abs(_knee_position(cn, H) - 0.3) < 0.1     # detrended knee is at the well


# --------------------------------------------------------------------------- #
# shared_knee: one knee polynomial tied across all terms
# --------------------------------------------------------------------------- #
def test_shared_knee_recovers_in_family_surface():
    """A surface whose opposite-slope term pair genuinely shares one (still
    T-dependent) knee is recovered as tightly as the unconstrained fit."""
    T, c, Tg, cg = _grid()
    knee = lambda t: -0.5 + 1e-4 * (t - 600)  # noqa: E731 -- identical for both terms
    amps = [lambda t: 0.05 + 2e-5 * (t - 600), lambda t: 0.03]
    slopes = [lambda t: 4.0 + 1e-3 * (t - 600), lambda t: -3.0 - 5e-4 * (t - 600)]
    H = _softplus_surface(T, c, amps=amps, slopes=slopes, knees=[knee, knee], offset=-0.2)

    surface = SoftplusSurface2DInterpolator(n_softplus=2, shared_knee=True).fit(T, c, H)

    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        Hq = _softplus_surface(Tq, cg, amps=amps, slopes=slopes, knees=[knee, knee], offset=-0.2)
        np.testing.assert_allclose(surface.slice_at(Tq)(cg), Hq, atol=RECOVER_ATOL)


def test_shared_knee_ties_terms_at_every_temperature():
    """With ``shared_knee=True`` every term's knee coincides exactly, not just at
    the training temperatures -- the fitted polynomial coefficients are tied, so
    this holds at any query T including extrapolated ones."""
    T, c, *_ = _grid()
    H = 0.4 * (c - 0.55) ** 2 + 2e-5 * (T - 600) * (c - 0.55) ** 2
    surface = SoftplusSurface2DInterpolator(n_softplus=3, shared_knee=True).fit(T, c, H)
    for Tq in (200.0, 600.0, 950.0):  # 950 is outside the (400, 800) training range
        knees = surface.slice_at(Tq).c
        np.testing.assert_array_equal(knees, np.full_like(knees, knees[0]))


def test_shared_knee_reduces_parameter_count():
    """Tying the knee across terms drops it from ``n_softplus`` independent
    polynomials to one, the only parameter-count difference between the two
    settings."""
    shared = SoftplusSurface2DInterpolator(n_softplus=3, c_order=2, shared_knee=True)
    independent = SoftplusSurface2DInterpolator(n_softplus=3, c_order=2, shared_knee=False)
    assert independent._n_params - shared._n_params == (3 - 1) * (2 + 1)


# --------------------------------------------------------------------------- #
# monotone_slope: |b_i(T)| never reverses (a convex well only sharpens as it cools)
# --------------------------------------------------------------------------- #
MONO_ATOL = 1e-9  # forward-difference slack for "|b| non-increasing in T"


def _abs_slopes_over_T(surface, Ts):
    """``|b_i(T)|`` for every term, stacked as ``(len(Ts), n_softplus)``."""
    return np.array([np.abs(surface.slice_at(float(t)).b) for t in Ts])


def test_monotone_slope_recovers_in_family_surface():
    """``_v_slope_1overT`` sharpens monotonically (b = 4 + 30/T), so the monotone
    parametrisation can still recover it tightly -- the constraint is consistent
    with the data and costs no accuracy."""
    Tg = np.linspace(200.0, 1000.0, 40)
    cg = np.linspace(0.30, 0.45, 11)
    T = np.repeat(Tg, len(cg))
    c = np.tile(cg, len(Tg))
    surface = SoftplusSurface2DInterpolator(n_softplus=2, monotone_slope=True).fit(
        T, c, _v_slope_1overT(T, c))
    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        np.testing.assert_allclose(surface.slice_at(Tq)(cg), _v_slope_1overT(Tq, cg), atol=1e-6)


def test_monotone_slope_is_monotone_far_outside_training_range():
    """``|b_i(T)|`` is non-increasing in T at *every* term and temperature,
    including heavy extrapolation on both sides of the (250, 1500) training range
    -- the clamp ``w = max(Wn - wmin, 0)`` holds it flat past the hot end rather
    than letting the even-power term turn it back up."""
    c0 = 0.36
    Tg = np.linspace(250.0, 1500.0, 50)
    cg = np.linspace(0.30, 0.45, 13)
    T = np.repeat(Tg, len(cg))
    c = np.tile(cg, len(Tg))

    def H(Tq, cq):
        Tq, u = np.asarray(Tq, float), np.asarray(cq, float) - c0
        sL, sR = 0.03 + 0.22 * (1000.0 / Tq), 0.01 + 0.09 * (1000.0 / Tq)
        return np.where(u < 0, sL * (-u), sR * u)

    surface = SoftplusSurface2DInterpolator(n_softplus=2, monotone_slope=True).fit(T, c, H(T, c))
    Ts = np.linspace(50.0, 5000.0, 300)  # far below and above the training window
    babs = _abs_slopes_over_T(surface, Ts)
    assert np.diff(babs, axis=0).max() <= MONO_ATOL


def test_monotone_slope_constrains_a_non_monotone_truth():
    """When the true sharpness is genuinely humped in 1/T, the free polynomial
    slope reproduces the reversal while ``monotone_slope`` refuses it -- proof the
    constraint binds rather than being vacuous."""
    c0 = 0.36

    def H(Tq, cq):
        w = 1000.0 / np.asarray(Tq, float)
        b = np.maximum(6.0 + 25.0 * w - 6.0 * w ** 2, 1.0)  # non-monotone in 1/T
        u = np.asarray(cq, float) - c0
        return -0.2 + 0.05 * _softplus(b * u) + 0.03 * _softplus(-b * u)

    Tg = np.linspace(250.0, 1500.0, 40)
    cg = np.linspace(0.30, 0.45, 13)
    T = np.repeat(Tg, len(cg))
    c = np.tile(cg, len(Tg))
    free = SoftplusSurface2DInterpolator(n_softplus=2, monotone_slope=False).fit(T, c, H(T, c))
    mono = SoftplusSurface2DInterpolator(n_softplus=2, monotone_slope=True).fit(T, c, H(T, c))

    Ts = np.linspace(Tg[0], Tg[-1], 150)
    assert _abs_slopes_over_T(free, Ts)[:, 0].max() - _abs_slopes_over_T(free, Ts)[:, 0][0] > 0
    assert np.diff(_abs_slopes_over_T(free, Ts), axis=0).max() > 1e-2   # free reverses
    assert np.diff(_abs_slopes_over_T(mono, Ts), axis=0).max() <= MONO_ATOL  # monotone does not


def test_monotone_slope_keeps_slices_convex():
    """The amplitude softplus link is untouched, so a monotone-slope fit on
    arbitrary noisy data still yields non-negative amplitudes and convex slices
    while honouring the slope constraint."""
    rng = np.random.default_rng(0)
    T, c, Tg, cg = _grid(nT=18, nc=9)
    H = rng.normal(scale=0.05, size=c.shape)
    surface = SoftplusSurface2DInterpolator(monotone_slope=True, max_nfev=300).fit(T, c, H)
    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        sl = surface.slice_at(Tq)
        assert (sl.a >= 0).all()
        assert _second_difference_min(sl, 0.2, 0.8) >= -CONVEX_ATOL
    Ts = np.linspace(Tg[0], Tg[-1], 120)
    assert np.diff(_abs_slopes_over_T(surface, Ts), axis=0).max() <= MONO_ATOL


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


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy"])
def test_robust_loss_produces_convex_fit(loss):
    """A non-default robust ``loss`` (for data with genuine outliers) must still
    produce a finite, convex slice that recovers a clean convex surface."""
    T, c, Tg, cg = _grid()
    H = 0.4 * (c - 0.5) ** 2 + 2e-5 * (T - 600) * (c - 0.5) ** 2
    surface = SoftplusSurface2DInterpolator(loss=loss, f_scale=0.5).fit(T, c, H)
    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        sl = surface.slice_at(Tq)
        pred = np.asarray(sl(cg))
        assert np.isfinite(pred).all()
        assert (sl.a >= 0).all()
        assert _second_difference_min(sl, 0.2, 0.8) >= -CONVEX_ATOL
        np.testing.assert_allclose(pred, H[np.isclose(T, Tq)], atol=5e-3)


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
    assert SoftplusSurface2DInterpolator(monotone_slope=True) != a
    with pytest.raises(Exception):
        a.n_softplus = 5


def test_method_override_selects_solver():
    """``method`` forces the coupled solver to ``'lm'`` or ``'trf'``; both recover
    a smooth convex surface (the version-dependent fragility is only for the stiff
    sharp-knee case), the field participates in equality, and ``'lm'`` rejects a
    robust loss up front."""
    T, c, Tg, cg = _grid()
    H = 0.4 * (c - 0.5) ** 2 + 2e-5 * (T - 600) * (c - 0.5) ** 2
    for method in ("lm", "trf"):
        surface = SoftplusSurface2DInterpolator(method=method).fit(T, c, H)
        for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
            np.testing.assert_allclose(surface.slice_at(Tq)(cg), H[np.isclose(T, Tq)], atol=2e-3)

    assert SoftplusSurface2DInterpolator(method="trf") != SoftplusSurface2DInterpolator()
    with pytest.raises(ValueError, match="loss='linear'"):
        SoftplusSurface2DInterpolator(method="lm", loss="soft_l1")


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


def test_standardize_centers_scales_and_guards_constant():
    """``_standardize`` centres on the mean and scales by the std, returning the
    shift/scale needed to reproduce the transform; a constant input falls back to
    scale 1 instead of dividing by zero."""
    x = np.array([400.0, 800.0, 1600.0])
    xn, shift, scale = _standardize(x)
    assert (shift, scale) == pytest.approx((x.mean(), x.std()))
    np.testing.assert_allclose(xn, (x - x.mean()) / x.std())
    # reapply the frozen shift/scale to a fresh input
    np.testing.assert_allclose((np.array([400.0]) - shift) / scale, (400.0 - x.mean()) / x.std())

    xn0, _, scale0 = _standardize(np.full(5, 7.0))
    assert scale0 == 1.0
    np.testing.assert_allclose(xn0, 0.0)
