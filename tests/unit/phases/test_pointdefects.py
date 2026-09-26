"""Direct tests for the saturation clamp of :class:`PointDefectedPhase`.

The unnormalised site fractions of a :class:`LowTemperatureExpansionSublattice`
leave the dilute regime with ``c`` outside ``[0, 1]`` and ``phi`` diverging to
``-inf``; ``_safe_dmu_bound`` / ``_saturation_window`` / ``_clamp_fixed_T`` are
what bound both.  The exact :class:`PointDefectSublattice` walks the trivial
path through all three (both crossings are ``+-inf``), so the public-interface
tests never exercise the clamp branches.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from landau.phases import LinePhase, kB
from landau.phases.pointdefects import (
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
