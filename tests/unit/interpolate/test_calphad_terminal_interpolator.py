import numpy as np, pytest
from landau.interpolate import (CalphadSurface2DInterpolator, SGTE, PolyFit, StitchedFit,
                                WhitneyTemperatureInterpolator)


def _surface(n_T=24, n_c=9, T_lo=900.0, T_hi=1300.0):
    """A liquid-like surface: large, temperature-dependent entropy, narrow T window."""
    Ts = np.linspace(T_lo, T_hi, n_T)
    cs = np.linspace(0.0, 1.0, n_c)
    T, C = np.meshgrid(Ts, cs, indexing="ij")
    kB = 8.617333262e-5
    S = 9.0 * kB + 2.0e-3 * kB * (T - T_lo)          # entropy grows with T
    H = -2.0 - 0.5 * C - T * S + C * (1 - C) * (-0.15 + 4e-5 * T)
    return T.ravel(), C.ravel(), H.ravel()


def test_terminal_interpolator_defaults_to_sgte():
    """Omitting the field must reproduce the previous behaviour exactly."""
    T, c, f = _surface()
    a = CalphadSurface2DInterpolator().fit(T, c, f)
    b = CalphadSurface2DInterpolator(terminal_interpolator=SGTE(4)).fit(T, c, f)
    g = np.linspace(950.0, 1250.0, 11)
    for Ti in g:
        assert a.slice_at(Ti)(0.5) == pytest.approx(b.slice_at(Ti)(0.5), rel=1e-12)


def test_whitney_terminal_keeps_entropy_bounded_below_the_data():
    """SGTE extrapolates with unbounded entropy; a linear low-side extension does not.

    Below the fitted window SGTE's S = -b - c(1 + ln T) decays without limit, which lets a
    liquid drop below the entropy of a competing solid and cross under it. StitchedFit with
    `low=PolyFit(2)` continues linearly in T, i.e. at constant entropy.
    """
    kB = 8.617333262e-5
    T, c, f = _surface()
    sg = CalphadSurface2DInterpolator().fit(T, c, f)
    st = CalphadSurface2DInterpolator(
        terminal_interpolator=WhitneyTemperatureInterpolator()).fit(T, c, f)

    def entropy(surf, Ti, ci=0.5, h=2.0):
        return -(surf.slice_at(Ti + h)(ci) - surf.slice_at(Ti - h)(ci)) / (2 * h) / kB

    in_window = entropy(sg, 1100.0)
    assert in_window == pytest.approx(entropy(st, 1100.0), rel=1e-6), "must agree inside the data"

    far_below = 400.0
    assert entropy(st, far_below) > 0.5 * in_window, "stitched entropy should stay near its edge value"
    assert entropy(st, far_below) > entropy(sg, far_below), "stitched must not decay as far as SGTE"


def test_concentration_structure_is_untouched():
    """Only the terminal T-model changes; the Redlich-Kister fit in c must be identical."""
    T, c, f = _surface()
    sg = CalphadSurface2DInterpolator().fit(T, c, f)
    st = CalphadSurface2DInterpolator(
        terminal_interpolator=WhitneyTemperatureInterpolator()).fit(T, c, f)
    Ti = 1100.0
    a = np.array([sg.slice_at(Ti)(x) for x in np.linspace(0.05, 0.95, 15)])
    b = np.array([st.slice_at(Ti)(x) for x in np.linspace(0.05, 0.95, 15)])
    assert np.allclose(a - a.mean(), b - b.mean(), atol=1e-9), "c-dependence must not change"


def test_stitched_terminal_is_an_equivalent_alternative():
    """StitchedFit reaches the same place; documented so the choice is informed, not arbitrary."""
    kB = 8.617333262e-5
    T, c, f = _surface()
    w = CalphadSurface2DInterpolator(
        terminal_interpolator=WhitneyTemperatureInterpolator()).fit(T, c, f)
    st = CalphadSurface2DInterpolator(
        terminal_interpolator=StitchedFit(interpolating=SGTE(4), low=PolyFit(2))).fit(T, c, f)
    ent = lambda s_, Ti, h=2.0: -(s_.slice_at(Ti + h)(0.5) - s_.slice_at(Ti - h)(0.5)) / (2 * h) / kB
    assert ent(w, 400.0) == pytest.approx(ent(st, 400.0), abs=1.0)
