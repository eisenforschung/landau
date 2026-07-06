"""Tests for the CALPHAD-style 2-D surface interpolator.

``CalphadSurface2DInterpolator`` fits the entropy-removed free energy as SGTE
terminals plus polynomial-in-T Redlich-Kister interaction coefficients, and
slices at fixed T to a :class:`RedlichKisterInterpolation` with an analytic
c-derivative.  The scenarios cover exact in-family recovery, the analytic
derivative, the terminal-phase requirement, the frozen-dataclass contract, and
end-to-end use inside ``Surface2DInterpolatingPhase`` (Gibbs-Duhem + convex f).
"""

import numpy as np
import pytest

from landau.interpolate import CalphadSurface2DInterpolator, SGTE
from landau.interpolate.basic import (
    CalphadFittedSurface,
    RedlichKisterInterpolation,
    SurfaceInterpolator,
)
from landau.phases import Surface2DInterpolatingPhase, TemperatureDependentLinePhase, S

RECOVER_ATOL = 1e-6   # exact in-family recovery (measured ~1e-10)
DERIV_ATOL = 1e-6     # analytic vs finite-difference c-derivative (measured ~5e-11)
GIBBS_ATOL = 5e-6     # c = -dphi/ddmu via the analytic slice derivative (measured ~8e-9)
CONVEX_ATOL = 1e-9    # second-difference convexity floor


def _calphad_surface(T, c, *, f0_lin, df_lin, L_lin):
    """Ground-truth entropy-removed H(T, c): SGTE(linear)-terminals + RK-in-c with
    each interaction coefficient a linear polynomial in T.

    ``f0_lin``/``df_lin`` are ``(a, b)`` for ``a + b*T``; ``L_lin`` is a list of
    such pairs, one per Redlich-Kister order.  Everything is representable by the
    interpolator's own model (SGTE order>=1, PolyFit order>=1), so recovery is
    exact up to solver precision.
    """
    T = np.asarray(T, float)
    c = np.asarray(c, float)
    f0 = f0_lin[0] + f0_lin[1] * T
    df = df_lin[0] + df_lin[1] * T
    H = (1 - c) * f0 + c * (f0 + df)
    pre = c * (1 - c)
    for v, (a, b) in enumerate(L_lin):
        H = H + pre * (a + b * T) * (1 - 2 * c) ** v
    return H


def _grid(Tlo=400.0, Thi=800.0, nT=25, nc=11):
    Tg = np.linspace(Tlo, Thi, nT)
    cg = np.linspace(0.0, 1.0, nc)  # terminals included, as Calphad requires
    return np.repeat(Tg, nc), np.tile(cg, nT), Tg, cg


# ground-truth coefficients reused across recovery/derivative tests
_F0 = (-0.5, 1e-4)
_DF = (0.3, -2e-4)
_L = [(0.1, 1e-4), (-0.05, 0.0)]


# --------------------------------------------------------------------------- #
# functional recovery
# --------------------------------------------------------------------------- #
def test_recovers_in_family_surface():
    """A surface that *is* SGTE-terminals + poly-in-T Redlich-Kister is recovered
    to solver precision at every temperature, terminals included."""
    T, c, Tg, cg = _grid()
    H = _calphad_surface(T, c, f0_lin=_F0, df_lin=_DF, L_lin=_L)
    surface = CalphadSurface2DInterpolator(num_coeffs=2, coeff_poly_order=1).fit(T, c, H)
    assert isinstance(surface, CalphadFittedSurface)
    for Tq in (Tg[0], Tg[len(Tg) // 2], Tg[-1]):
        Hq = _calphad_surface(Tq, cg, f0_lin=_F0, df_lin=_DF, L_lin=_L)
        np.testing.assert_allclose(surface.slice_at(Tq)(cg), Hq, atol=RECOVER_ATOL)


def test_slice_is_redlich_kister_with_analytic_derivative():
    """Each fixed-T slice is a :class:`RedlichKisterInterpolation` whose analytic
    c-derivative matches a central difference of the slice."""
    T, c, Tg, cg = _grid()
    H = _calphad_surface(T, c, f0_lin=_F0, df_lin=_DF, L_lin=_L)
    surface = CalphadSurface2DInterpolator(num_coeffs=2, coeff_poly_order=1).fit(T, c, H)
    cc = np.linspace(0.05, 0.95, 50)
    h = 1e-6
    for Tq in (450.0, 650.0, 790.0):
        sl = surface.slice_at(Tq)
        assert isinstance(sl, RedlichKisterInterpolation)
        fd = (sl(cc + h) - sl(cc - h)) / (2 * h)
        np.testing.assert_allclose(sl.deriv()(cc), fd, atol=DERIV_ATOL)


def test_slice_temperature_dependence_is_real():
    """Distinct temperatures give distinct slices, each matching its own data."""
    T, c, Tg, cg = _grid()
    H = _calphad_surface(T, c, f0_lin=_F0, df_lin=_DF, L_lin=_L)
    surface = CalphadSurface2DInterpolator(num_coeffs=2, coeff_poly_order=1).fit(T, c, H)
    lo = np.asarray(surface.slice_at(400.0)(cg))
    hi = np.asarray(surface.slice_at(800.0)(cg))
    assert np.abs(hi - lo).max() > 1e-2
    np.testing.assert_allclose(lo, _calphad_surface(400.0, cg, f0_lin=_F0, df_lin=_DF, L_lin=_L), atol=RECOVER_ATOL)
    np.testing.assert_allclose(hi, _calphad_surface(800.0, cg, f0_lin=_F0, df_lin=_DF, L_lin=_L), atol=RECOVER_ATOL)


# --------------------------------------------------------------------------- #
# shape contract
# --------------------------------------------------------------------------- #
def test_slice_scalar_and_array_shapes():
    T, c, Tg, cg = _grid()
    H = _calphad_surface(T, c, f0_lin=_F0, df_lin=_DF, L_lin=_L)
    sl = CalphadSurface2DInterpolator(num_coeffs=2).fit(T, c, H).slice_at(600.0)

    scalar = sl(0.5)
    assert np.ndim(scalar) == 0
    arr = sl(np.linspace(0.1, 0.9, 11))
    assert isinstance(arr, np.ndarray) and arr.shape == (11,)
    assert np.ndim(sl.deriv()(0.5)) == 0


# --------------------------------------------------------------------------- #
# input validation, immutability
# --------------------------------------------------------------------------- #
def test_fit_requires_terminals():
    """Without concentrations reaching c=0 and c=1 the fit raises -- the SGTE
    terminals are undefined otherwise."""
    Tg = np.linspace(400, 800, 10)
    cg = np.linspace(0.2, 0.8, 7)  # no terminals
    T = np.repeat(Tg, len(cg))
    c = np.tile(cg, len(Tg))
    f = _calphad_surface(T, c, f0_lin=_F0, df_lin=_DF, L_lin=_L)
    with pytest.raises(ValueError, match="terminal concentrations at c=0 and c=1"):
        CalphadSurface2DInterpolator().fit(T, c, f)


def test_interpolator_is_frozen_and_hashable():
    a = CalphadSurface2DInterpolator()
    b = CalphadSurface2DInterpolator()
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1
    assert CalphadSurface2DInterpolator(num_coeffs=3) != a
    assert CalphadSurface2DInterpolator(coeff_poly_order=3) != a
    assert CalphadSurface2DInterpolator(terminal_sgte_order=5) != a
    assert isinstance(a, SurfaceInterpolator)
    with pytest.raises(Exception):
        a.num_coeffs = 7


# --------------------------------------------------------------------------- #
# integration with Surface2DInterpolatingPhase
# --------------------------------------------------------------------------- #
def _solution_line_phases(ncomp=9):
    """Full [0, 1] solution phase with a convex (ordering) excess, terminals
    included so the Calphad surface can fit.  Free energy stored WITH entropy;
    the phase is built ``add_entropy=False`` so the surface sees H = f + T*S."""
    Tsweep = np.linspace(400.0, 800.0, 60)
    cs = np.linspace(0.0, 1.0, ncomp)
    lines = []
    for ci in cs:
        # negative regular parameter -> convex f(c), so c sweeps the interior
        H = -0.3 * ci * (1 - ci) + (-0.5 + 1e-4 * Tsweep) * (1 - ci) + (-0.2 - 5e-5 * Tsweep) * ci
        f = H - Tsweep * S(np.array(ci))
        lines.append(TemperatureDependentLinePhase("sol", float(ci), Tsweep, f, interpolator=SGTE(3)))
    return lines


def test_surface_phase_solver_and_gibbs_duhem():
    phase = Surface2DInterpolatingPhase(
        "sol", _solution_line_phases(), concentration_range=(0.0, 1.0),
        add_entropy=False, temperature_range=(400.0, 800.0),
        surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=3))

    T = 500.0
    dmu = np.linspace(-0.6, 0.4, 61)
    c = phase.concentration(np.full_like(dmu, T), dmu)
    phi = phase.semigrand_potential(np.full_like(dmu, T), dmu)
    assert np.isfinite(c).all() and np.isfinite(phi).all()
    assert (c >= -1e-6).all() and (c <= 1.0 + 1e-6).all()

    h = 1e-4
    cphi = -(phase.semigrand_potential(np.full_like(dmu, T), dmu + h)
             - phase.semigrand_potential(np.full_like(dmu, T), dmu - h)) / (2 * h)
    interior = (c > 0.05) & (c < 0.95)
    assert interior.sum() > 10
    np.testing.assert_allclose(c[interior], cphi[interior], atol=GIBBS_ATOL)


def test_surface_phase_free_energy_is_convex():
    """The ordering excess plus ideal entropy gives a convex f(c): no spurious
    miscibility gap across the window and temperature."""
    phase = Surface2DInterpolatingPhase(
        "sol", _solution_line_phases(), concentration_range=(0.0, 1.0),
        add_entropy=False, temperature_range=(400.0, 800.0),
        surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=3))
    cc = np.linspace(0.02, 0.98, 150)
    for T in (450.0, 600.0, 750.0):
        f = phase.free_energy(T, cc)
        d2 = f[2:] - 2 * f[1:-1] + f[:-2]
        assert d2.min() >= -CONVEX_ATOL
