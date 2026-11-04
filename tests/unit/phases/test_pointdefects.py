"""Direct tests for the saturation clamp of :class:`PointDefectedPhase`.

The unnormalised site fractions of a :class:`LowTemperatureExpansionSublattice`
leave the dilute regime with ``c`` outside ``[0, 1]`` and ``phi`` diverging to
``-inf``; ``_safe_dmu_bound`` / ``_saturation_window`` / ``_clamp_fixed_T`` are
what bound both.  The exact :class:`PointDefectSublattice` walks the trivial
path through all three (both crossings are ``+-inf``), so the public-interface
tests never exercise the clamp branches.

The :class:`CappedPointDefectSublattice` block below pins the third sibling,
bounded like the exact model but pinned at a total defect site fraction ``m``
past the cap ``s* = m / (1 - m)``.
"""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from landau.phases import LinePhase, kB
from landau.phases.pointdefects import (
    CappedPointDefectSublattice,
    ConstantPointDefect,
    LowTemperatureExpansionSublattice,
    PointDefectedPhase,
    PointDefectSublattice,
)

# the crossings are located by brentq on its default xtol of 2e-12 in dmu
_CROSSING_ATOL = 1e-9
# well inside the window at the fixture's energies, and the same T throughout so
# the crossings every test reads back are one pair
_T = 800.0


def _b2_phase(sublattice_class):
    """B2 phase with an antisite and a vacancy on each of the two sublattices.

    The two antisites differ in energy on purpose: at equal energies
    ``c(-dmu) = 1 - c(dmu)``, which makes the two saturation crossings exact
    mirror images and hides anything that derives one from the other.
    """
    host = LinePhase("AB", fixed_concentration=0.5, line_energy=-0.40, line_entropy=1.0 * kB)
    alpha = sublattice_class(
        name="alpha",
        sublattice=0,
        sublattice_fraction=0.5,
        defects=[
            ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1),
            ConstantPointDefect("V_a", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
        ],
    )
    beta = sublattice_class(
        name="beta",
        sublattice=1,
        sublattice_fraction=0.5,
        defects=[
            ConstantPointDefect("A_b", excess_energy=0.45, excess_entropy=0.0, excess_solutes=-1),
            ConstantPointDefect("V_b", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
        ],
    )
    return PointDefectedPhase(name="AB", line_phase=host, sublattices=[alpha, beta])


# --- _safe_dmu_bound ---


@pytest.mark.parametrize("T", [300.0, 800.0, 1500.0])
def test_safe_dmu_bound_does_not_overflow(T):
    """Evaluating the raw partition variables at ``+-bound`` stays finite -- that
    is the whole point of the bound, since ``_saturation_window`` brackets on it."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    M = phase._safe_dmu_bound(T)
    for sub in phase.sublattices:
        assert np.all(np.isfinite(sub._get_zes(T, np.array([-M, M]))))


def test_safe_dmu_bound_grows_with_temperature():
    """A hotter phase tolerates a wider ``dmu`` bracket before ``exp`` overflows."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    bounds = [phase._safe_dmu_bound(T) for T in (300.0, 800.0, 1500.0)]
    assert bounds[0] < bounds[1] < bounds[2]


def test_safe_dmu_bound_falls_back_to_the_floor_below_the_thermal_budget():
    """At 5 K the formation energies exceed ``_EXP_LIMIT * kB * T``, so the
    expression goes negative and the bound is the floor itself -- without it the
    bracket would come out reversed."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    assert phase._safe_dmu_bound(5.0) == 1e-6


# --- _saturation_window ---


@pytest.mark.parametrize("T", [200.0, 1000.0, 4000.0])
def test_saturation_window_inert_for_the_exact_model(T):
    """Site competition keeps ``c`` inside ``(0, 1)``, so there is nothing to
    clamp and the window is unbounded at any temperature."""
    phase = _b2_phase(PointDefectSublattice)
    assert phase._saturation_window(T) == (-np.inf, np.inf)


def test_saturation_window_locates_the_lte_crossings():
    """For the unbounded model both crossings are finite and the raw
    concentration is exactly 0 / 1 there."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    dmu_lo, dmu_hi = phase._saturation_window(_T)
    assert np.isfinite(dmu_lo) and np.isfinite(dmu_hi)
    assert dmu_lo < dmu_hi
    assert phase._raw_phi_c(_T, dmu_lo)[1] == pytest.approx(0.0, abs=_CROSSING_ATOL)
    assert phase._raw_phi_c(_T, dmu_hi)[1] == pytest.approx(1.0, abs=_CROSSING_ATOL)


def test_saturation_window_reports_no_crossing_outside_the_safe_bound():
    """At 5 K the formation energies exceed ``_EXP_LIMIT * kB * T``, so the
    bracket collapses below the crossings; that reports as no crossing instead
    of raising out of ``brentq``."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    assert phase._saturation_window(5.0) == (-np.inf, np.inf)


# --- _clamp_fixed_T ---


def test_clamp_leaves_the_interior_untouched():
    """Inside the window the clamp returns the raw values unchanged, save for
    ``c`` being held inside ``[0, 1]`` at the crossings themselves."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    dmu_lo, dmu_hi = phase._saturation_window(_T)
    dmu = np.linspace(dmu_lo, dmu_hi, 11)
    phi, c = phase._clamp_fixed_T(_T, dmu)
    raw_phi, raw_c = phase._raw_phi_c(_T, dmu)
    assert_array_equal(phi, raw_phi)
    assert_array_equal(c, np.clip(raw_c, 0.0, 1.0))


@pytest.mark.parametrize("T", [100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1000.0, 1500.0])
def test_clamp_keeps_c_in_the_unit_interval_up_to_the_crossings(T):
    """``brentq`` stops within ``xtol`` of each crossing on either side of it, so
    the raw concentration at the located crossing can sit a few 1e-12 past 0 or
    1 -- worst at low T, where ``dc/d(dmu)`` scales like ``1/(kB T)``. Which side
    it lands on varies with T; the range is exact regardless."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    dmu_lo, dmu_hi = phase._saturation_window(T)
    dmu = np.linspace(dmu_lo, dmu_hi, 1001)
    _, c = phase._clamp_fixed_T(T, dmu)
    assert np.all((c >= 0.0) & (c <= 1.0))


def test_clamp_above_saturation_is_a_line_phase_at_c_one():
    """Past the upper crossing the phase sits at ``c = 1`` and ``phi`` continues
    with slope ``-1``, anchored on the raw value at the crossing."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    _, dmu_hi = phase._saturation_window(_T)
    dmu = dmu_hi + np.array([1e-6, 0.1, 0.5])
    phi, c = phase._clamp_fixed_T(_T, dmu)
    assert_array_equal(c, 1.0)
    assert_allclose(phi, phase._raw_phi_c(_T, dmu_hi)[0] - (dmu - dmu_hi), rtol=1e-14, atol=0)


def test_clamp_below_saturation_is_a_line_phase_at_c_zero():
    """Past the lower crossing the phase sits at ``c = 0`` and ``phi`` is flat at
    the raw value there."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    dmu_lo, _ = phase._saturation_window(_T)
    dmu = dmu_lo - np.array([1e-6, 0.1, 0.5])
    phi, c = phase._clamp_fixed_T(_T, dmu)
    assert_array_equal(c, 0.0)
    assert_allclose(phi, phase._raw_phi_c(_T, dmu_lo)[0], rtol=1e-14, atol=0)


def test_clamp_applies_every_rule_in_one_array_call():
    """A single call mixing saturated and interior chemical potentials gets each
    region's rule, independent of the order the points arrive in."""
    phase = _b2_phase(LowTemperatureExpansionSublattice)
    dmu_lo, dmu_hi = phase._saturation_window(_T)
    interior = 0.5 * (dmu_lo + dmu_hi)
    dmu = np.array([dmu_hi + 0.2, dmu_lo - 0.2, interior])
    phi, c = phase._clamp_fixed_T(_T, dmu)
    assert_allclose(c, [1.0, 0.0, phase._raw_phi_c(_T, interior)[1]], rtol=1e-14, atol=0)
    expected_phi = [
        phase._raw_phi_c(_T, dmu_hi)[0] - 0.2,
        phase._raw_phi_c(_T, dmu_lo)[0],
        phase._raw_phi_c(_T, interior)[0],
    ]
    assert_allclose(phi, expected_phi, rtol=1e-14, atol=0)


# --- CappedPointDefectSublattice ---

_M = 0.1
_S_STAR = _M / (1 - _M)
_H = -_M * np.log(_M) - (1 - _M) * np.log(1 - _M)


def _capped(m=_M):
    return partial(CappedPointDefectSublattice, max_site_fraction=m)


def _single_defect_sublattice(sublattice_class, defects=None, eta=0.5):
    defects = defects or [ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1)]
    return sublattice_class(name="alpha", sublattice=0, sublattice_fraction=eta, defects=defects)


def _zes_sum(sublattice, T, dmu):
    """Independent ``s = sum_i exp(-([g_i] - [n_i] dmu) / kB T)``."""
    return sum(np.exp(-(d.excess_free_energy(T) - d.excess_solutes * dmu) / kB / T) for d in sublattice.defects)


def test_capped_matches_exact_below_the_cap():
    """Below ``s*`` the capped model is the exact one, contribution by contribution."""
    capped, exact = _b2_phase(_capped()), _b2_phase(PointDefectSublattice)
    dmu = np.linspace(-0.25, 0.1, 201)
    for sub_c, sub_e in zip(capped.sublattices, exact.sublattices):
        assert np.all(_zes_sum(sub_c, _T, dmu) < _S_STAR)  # the grid is wholly below the cap
        assert_allclose(
            sub_c.semigrand_potential_contribution(_T, dmu),
            sub_e.semigrand_potential_contribution(_T, dmu),
            rtol=0,
            atol=1e-12,
        )
        assert_allclose(
            sub_c.concentration_contribution(_T, dmu), sub_e.concentration_contribution(_T, dmu), rtol=0, atol=1e-12
        )


def test_capped_total_site_fraction_is_pinned_above_the_cap():
    """Above ``s*`` the sublattice is a line phase: the defect site fraction is ``m``
    whatever the chemical potential."""
    sub = _single_defect_sublattice(_capped())
    dmu = np.linspace(0.2, 2.0, 101)
    assert np.all(_zes_sum(sub, _T, dmu) > _S_STAR)  # the grid is wholly above the cap
    n = sub.defects[0].excess_solutes
    x = sub.concentration_contribution(_T, dmu) / (sub.sublattice_fraction * n)
    assert_allclose(x, _M, rtol=1e-12, atol=0)


def test_capped_log_partition_and_its_slope_are_continuous_at_the_cap():
    """Both branches meet at ``L(s*) = -ln(1 - m)`` with slope ``1 - m``."""
    sub = _single_defect_sublattice(_capped())
    L = sub._site_log_partition
    assert L(_S_STAR) == pytest.approx(-np.log(1 - _M), rel=1e-14)
    # upper branch at s*, written out
    assert _H + _M * np.log(_S_STAR) == pytest.approx(-np.log(1 - _M), rel=1e-14)
    assert L(_S_STAR * (1 + 1e-12)) == pytest.approx(-np.log(1 - _M), rel=1e-11)
    h = 1e-6 * _S_STAR
    left = (L(_S_STAR) - L(_S_STAR - h)) / h
    right = (L(_S_STAR + h) - L(_S_STAR)) / h
    assert left == pytest.approx(1 - _M, abs=1e-6)
    assert right == pytest.approx(1 - _M, abs=1e-6)


def test_capped_log_partition_curvature_jumps_at_the_cap():
    """``L''`` is ``-(1 - m)^2`` just below ``s*`` and ``-(1 - m)^2 / m`` just above,
    which is what makes ``dc/d(dmu)`` discontinuous."""
    sub = _single_defect_sublattice(_capped())
    L = sub._site_log_partition
    h = 1e-4 * _S_STAR
    left = (L(_S_STAR) - 2 * L(_S_STAR - h) + L(_S_STAR - 2 * h)) / h**2
    right = (L(_S_STAR + 2 * h) - 2 * L(_S_STAR + h) + L(_S_STAR)) / h**2
    assert left == pytest.approx(-((1 - _M) ** 2), rel=1e-3)
    assert right == pytest.approx(-((1 - _M) ** 2) / _M, rel=1e-3)


def test_capped_normalization_is_the_inverse_slope():
    """``D = 1 / L'`` -- what ``c = -d(phi)/d(dmu)`` demands -- on both branches."""
    sub = _single_defect_sublattice(_capped())
    s = np.geomspace(1e-3, 1e3, 401)
    h = 1e-6 * s
    s, h = s[np.abs(s - _S_STAR) > 2 * h], h[np.abs(s - _S_STAR) > 2 * h]
    assert s.min() < _S_STAR < s.max()  # straddles the cap
    L = sub._site_log_partition
    dL = (L(s + h) - L(s - h)) / (2 * h)
    assert_allclose(sub._site_normalization(s) * dL, 1.0, rtol=0, atol=1e-8)


def test_capped_budget_is_split_by_boltzmann_weight():
    """Two defects of different ``[n]`` and ``[g]``: above the cap each gets
    ``x_i = z_i m / s`` and together they fill the budget ``m`` exactly."""
    defects = [
        ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1),
        ConstantPointDefect("BB_a", excess_energy=0.55, excess_entropy=0.0, excess_solutes=+2),
    ]
    sub = _single_defect_sublattice(_capped(), defects=defects)
    dmu = np.linspace(0.2, 0.8, 61)
    z = np.array([np.exp(-(d.excess_energy - d.excess_solutes * dmu) / kB / _T) for d in defects])
    s = z.sum(axis=0)
    assert np.all(s > _S_STAR)
    # both defects carry a non-negligible share somewhere on the grid
    assert (z[1] / s).max() > 0.5 and (z[0] / s).max() > 0.5
    zes = sub._get_zes(_T, dmu)
    x = zes / sub._site_normalization(zes.sum(axis=0))
    assert_allclose(x, z * _M / s, rtol=1e-12, atol=0)
    assert_allclose(x.sum(axis=0), _M, rtol=1e-12, atol=0)
    n = np.array([d.excess_solutes for d in defects])
    assert_allclose(
        sub.concentration_contribution(_T, dmu),
        sub.sublattice_fraction * (n[:, None] * z * _M / s).sum(axis=0),
        rtol=1e-12,
        atol=0,
    )


def test_capped_phase_concentration_is_minus_dphi_ddmu_across_the_kinks():
    """``c = -d(phi)/d(dmu)`` holds through both kinks of the full B2 phase."""
    phase = _b2_phase(_capped())
    dmu = np.linspace(-0.6, 0.5, 2001)
    for sub in phase.sublattices:  # each sublattice crosses its cap on the grid
        s = _zes_sum(sub, _T, dmu)
        assert s.min() < _S_STAR < s.max()
    c = phase.concentration(_T, dmu)
    h = 1e-6
    fd = -(phase.semigrand_potential(_T, dmu + h) - phase.semigrand_potential(_T, dmu - h)) / (2 * h)
    assert_allclose(fd, c, rtol=0, atol=1e-8)


def test_capped_phase_never_undercuts_the_exact_phase():
    """``phi_capped >= phi_exact`` everywhere and strictly so past the cap, so the
    capped phase cannot spuriously win an argmin over ``phi``."""
    capped, exact = _b2_phase(_capped()), _b2_phase(PointDefectSublattice)
    dmu = np.linspace(-1.0, 1.0, 801)
    gap = capped.semigrand_potential(_T, dmu) - exact.semigrand_potential(_T, dmu)
    assert gap.min() >= -1e-15
    assert gap.max() > 1e-3


@pytest.mark.parametrize("T", [100.0, 300.0, 800.0, 1500.0, 4000.0])
def test_capped_phase_needs_no_clamp(T):
    """Like the exact model, the capped one is bounded: the window is unbounded and
    ``c`` stays within ``0.5 +- eta m`` over the whole range the clamp probes."""
    phase = _b2_phase(_capped())
    assert phase._saturation_window(T) == (-np.inf, np.inf)
    M = phase._safe_dmu_bound(T)
    c = phase.concentration(T, np.linspace(-M, M, 4001))
    assert np.all(np.isfinite(c))
    assert np.all((c >= 0.5 - 0.5 * _M - 1e-12) & (c <= 0.5 + 0.5 * _M + 1e-12))


def test_capped_reproduces_the_exact_model_as_the_cap_approaches_one():
    """With ``m = 1 - eps`` past ``s* ~ 1/eps`` the two differ by at most
    ``eta kB T eps (1 + ln s)`` in ``phi`` and ``eta eps |[n]|`` in ``c`` per sublattice."""
    eps = 1e-9
    T = 1000.0
    capped, exact = _b2_phase(_capped(1 - eps)), _b2_phase(PointDefectSublattice)
    dmu = np.linspace(-2.5, 2.5, 501)
    s_max = max(_zes_sum(sub, T, dmu).max() for sub in capped.sublattices)
    assert s_max > (1 - eps) / eps  # the grid goes past the cap
    assert_allclose(
        capped.semigrand_potential(T, dmu),
        exact.semigrand_potential(T, dmu),
        rtol=0,
        atol=kB * T * eps * (1 + np.log(s_max)),
    )
    assert_allclose(capped.concentration(T, dmu), exact.concentration(T, dmu), rtol=0, atol=eps)


@pytest.mark.parametrize("m", [0.0, 1.0, -0.1, 1.5])
def test_capped_rejects_a_cap_outside_the_open_unit_interval(m):
    with pytest.raises(ValueError, match="max_site_fraction"):
        _single_defect_sublattice(_capped(m))


def _blocked_sublattice(defects, eta=0.5, m=_M):
    """Block exclusion as an exact sublattice on ``eta m`` sites (one per block of
    ``1/m``), every defect entropy raised by ``kB ln(1/m)`` (its ``1/m`` positions)."""
    return PointDefectSublattice(
        name="alpha",
        sublattice=0,
        sublattice_fraction=eta * m,
        defects=[replace(d, excess_entropy=d.excess_entropy + kB * np.log(1 / m)) for d in defects],
    )


_BLOCK_DEFECTS = [
    ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1),
    ConstantPointDefect("V_a", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
]


def test_block_exclusion_from_a_rescaled_exact_sublattice():
    """The rescaled exact sublattice gives ``L = m ln(1 + s/m)``, ``D = 1 + s/m`` per
    original site."""
    eta = 0.5
    blocked = _blocked_sublattice(_BLOCK_DEFECTS, eta=eta)
    reference = _single_defect_sublattice(PointDefectSublattice, defects=_BLOCK_DEFECTS, eta=eta)
    dmu = np.linspace(-0.5, 1.0, 301)
    s = _zes_sum(reference, _T, dmu)
    assert s.min() < 1e-2 * _M and s.max() > 1e3 * _M  # dilute through saturated
    assert_allclose(
        blocked.semigrand_potential_contribution(_T, dmu),
        -eta * kB * _T * _M * np.log1p(s / _M),
        rtol=1e-12,
        atol=0,
    )
    n = np.array([d.excess_solutes for d in _BLOCK_DEFECTS])
    z = np.array([np.exp(-(d.excess_energy - d.excess_solutes * dmu) / kB / _T) for d in _BLOCK_DEFECTS])
    assert_allclose(
        blocked.concentration_contribution(_T, dmu),
        eta * (n[:, None] * z).sum(axis=0) / (1 + s / _M),
        rtol=1e-12,
        atol=0,
    )


def test_block_exclusion_never_undercuts_the_exact_sublattice():
    """``m ln(1 + s/m) <= ln(1 + s)``: the blocked ``phi`` contribution is never below
    the exact one, and strictly above it once defects are present."""
    blocked = _blocked_sublattice(_BLOCK_DEFECTS)
    exact = _single_defect_sublattice(PointDefectSublattice, defects=_BLOCK_DEFECTS)
    dmu = np.linspace(-1.0, 1.0, 401)
    gap = blocked.semigrand_potential_contribution(_T, dmu) - exact.semigrand_potential_contribution(_T, dmu)
    assert gap.min() > 0
